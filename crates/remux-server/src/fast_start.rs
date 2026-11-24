use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, RwLock};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Redirect, Response};
use bytes::{Bytes, BytesMut};
use futures_util::StreamExt as _;
use serde_json::json;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Default head buffer size in bytes for standard 1080p / smaller streams: 32 MB.
pub const FAST_START_DEFAULT_HEAD_SIZE: usize = 32 * 1024 * 1024;

/// 4K / UHD / large stream head buffer size in bytes: 64 MB.
pub const FAST_START_4K_HEAD_SIZE: usize = 64 * 1024 * 1024;

/// Backward-compatible alias for default head size.
pub const FAST_START_HEAD_SIZE: usize = FAST_START_DEFAULT_HEAD_SIZE;

/// Tail buffer size for MKV Cues / seekhead index: 2 MB.
pub const FAST_START_TAIL_SIZE: usize = 2 * 1024 * 1024;

/// Head buffer size in bytes for standard 1080p Continue Watching / resume streams: 24 MB.
/// Provides plenty of headroom (~25s of 1080p bitstream) to parse container headers and tracks.
pub const FAST_START_RESUME_HEAD_SIZE_1080P: usize = 24 * 1024 * 1024;

/// Head buffer size in bytes for 4K / UHD Continue Watching / resume streams: 64 MB.
/// Provides sufficient headroom (~16s of 33 Mbps bitstream) for high-bitrate Dolby Vision/Atmos
/// files to complete avformat_find_stream_info entirely in RAM before seeking.
pub const FAST_START_RESUME_HEAD_SIZE_4K: usize = 64 * 1024 * 1024;

/// Backward-compatible alias for 4K resume head size.
pub const FAST_START_RESUME_HEAD_SIZE: usize = FAST_START_RESUME_HEAD_SIZE_4K;

/// Maximum chunk size in bytes served over LAN per head range request: 16 MB.
/// Remux buffers up to 64 MB in server RAM from WAN Debrid CDN to prevent probe eviction,
/// but serving in 16 MB chunks over LAN ensures low-bitrate titles and standard resumes
/// only transfer ~16 MB across the client's Wi-Fi link before seeking, while 4K titles
/// requiring more probe bytes seamlessly fetch subsequent 16 MB slices directly from RAM.
pub const FAST_START_HEAD_SERVE_CHUNK_SIZE: usize = 16 * 1024 * 1024;

/// Resume buffer size in bytes for standard 1080p Continue Watching streams: 48 MB.
pub const FAST_START_RESUME_BUFFER_SIZE_1080P: usize = 56 * 1024 * 1024;

/// Resume buffer size in bytes for 4K / UHD Continue Watching streams: 104 MB.
/// Provides ~25-30 seconds of forward 4K video playback while absorbing up to 20 seconds of Infuse auto-rewind.
pub const FAST_START_RESUME_BUFFER_SIZE_4K: usize = 104 * 1024 * 1024;

/// Backward-compatible alias for 4K resume buffer size.
pub const FAST_START_RESUME_BUFFER_SIZE: usize = FAST_START_RESUME_BUFFER_SIZE_4K;

/// Pre-roll padding in bytes for standard 1080p resume buffer: 20 MB.
/// Absorbs up to 20 seconds of Infuse auto-rewind on 1080p streams (~15-20s of demuxer backward headroom)
/// while leaving 36 MB (~35s of 1080p video) for smooth forward playback.
pub const FAST_START_RESUME_PREROLL_BYTES_1080P: usize = 20 * 1024 * 1024;

/// Pre-roll padding in bytes for 4K resume buffer: 48 MB.
/// Absorbs up to 20 seconds of Infuse auto-rewind on high-bitrate 4K streams (~18-20s of demuxer backward headroom)
/// while leaving 56 MB (~25-30s of 4K video) for smooth forward playback.
pub const FAST_START_RESUME_PREROLL_BYTES_4K: usize = 48 * 1024 * 1024;

/// Backward-compatible alias for 4K resume preroll bytes.
pub const FAST_START_RESUME_PREROLL_BYTES: usize = FAST_START_RESUME_PREROLL_BYTES_4K;

/// Global toggle: when enabled, Remux does NOT buffer head/tail bytes in RAM.
/// Instead, pre-warming only resolves the direct Debrid CDN edge URL, and playback requests
/// (from byte 0 to any seek) immediately receive an HTTP 307 redirect directly to the Debrid CDN.
pub static FORCE_DIRECT_307: LazyLock<AtomicBool> = LazyLock::new(|| {
    let enabled = std::env::var("REMUX_FORCE_DIRECT_307")
        .or_else(|_| std::env::var("FAST_START_FORCE_DIRECT_307"))
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    AtomicBool::new(enabled)
});

pub fn is_force_direct_307() -> bool {
    FORCE_DIRECT_307.load(Ordering::Relaxed)
}

pub fn set_force_direct_307(val: bool) {
    FORCE_DIRECT_307.store(val, Ordering::Relaxed);
}

/// Helper to determine if a stream is 4K / UHD based on URL clues, file size, or 4K flags.
pub fn is_4k_stream(
    url: &str,
    size_bytes: Option<u64>,
    is_4k: Option<bool>,
) -> bool {
    if is_4k == Some(true) {
        return true;
    }
    if let Some(bytes) = size_bytes {
        if bytes > 8 * 1024 * 1024 * 1024 {
            return true;
        }
    }
    let lower = url.to_ascii_lowercase();
    lower.contains("2160p")
        || lower.contains("4k")
        || lower.contains("uhd")
        || lower.contains("remux")
}

/// Determines the ideal Fast-Start head buffer size based on URL clues, file size, or 4K flags.
pub fn calculate_fast_start_head_size(
    url: &str,
    size_bytes: Option<u64>,
    is_4k: Option<bool>,
) -> usize {
    if is_4k_stream(url, size_bytes, is_4k) {
        FAST_START_4K_HEAD_SIZE
    } else {
        FAST_START_DEFAULT_HEAD_SIZE
    }
}

/// Determines the ideal Fast-Start resume head buffer size (24 MB for 1080p, 64 MB for 4K).
pub fn calculate_fast_start_resume_head_size(
    url: &str,
    size_bytes: Option<u64>,
    is_4k: Option<bool>,
) -> usize {
    if is_4k_stream(url, size_bytes, is_4k) {
        FAST_START_RESUME_HEAD_SIZE_4K
    } else {
        FAST_START_RESUME_HEAD_SIZE_1080P
    }
}

/// Maximum number of items held in memory simultaneously.
pub const FAST_START_MAX_ITEMS: usize = 50;

/// Default liveness re-verification threshold (1 hour).
pub const FAST_START_TTL: Duration = Duration::from_secs(60 * 60);

/// Cached fast-start entry for an upstream stream.
#[derive(Clone, Debug)]
pub struct FastStartEntry {
    pub item_id: Option<Uuid>,
    pub original_url: String,
    pub direct_cdn_url: String,
    pub total_size: u64,
    pub head_bytes: Bytes,
    pub tail_bytes: Option<Bytes>,
    pub resume_buffer: Option<(u64, Bytes)>,
    pub content_type: String,
    pub created_at: Instant,
}

/// In-flight pre-warm entry allowing clients to stream chunks in real-time over LAN before download completes.
pub struct InFlightHead {
    pub item_id: Option<Uuid>,
    pub original_url: String,
    pub direct_cdn_url: String,
    pub total_size: u64,
    pub target_head_size: usize,
    pub content_type: String,
    pub buffer: RwLock<BytesMut>,
    pub tail_bytes: RwLock<Option<Bytes>>,
    pub notify: tokio::sync::Notify,
    pub tail_notify: tokio::sync::Notify,
    pub is_done: AtomicBool,
    pub has_error: AtomicBool,
}

/// Priority lane for Fast-Start pre-warming.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrewarmPriority {
    /// Interactive / foreground user action (`PlaybackInfo`, direct `/stream` hit).
    /// Bypasses `FAST_START_PREWARM_SEMAPHORE` so the download starts in 0ms.
    High,
    /// Speculative background pre-warm (home screen, season browsing, next-episode).
    /// Waits for a permit from `FAST_START_PREWARM_SEMAPHORE` before locking `PREWARM_LOCKS`.
    Low,
}

/// Global metrics for fast-start acceleration.
pub struct FastStartMetrics {
    pub hits_head: AtomicU64,
    pub hits_tail: AtomicU64,
    pub hits_resume: AtomicU64,
    pub redirects_issued: AtomicU64,
    pub misses: AtomicU64,
    pub prewarms_started: AtomicU64,
    pub prewarms_high_started: AtomicU64,
    pub prewarms_low_started: AtomicU64,
    pub prewarms_succeeded: AtomicU64,
    pub prewarms_failed: AtomicU64,
    pub bytes_served_lan: AtomicU64,
}

impl FastStartMetrics {
    const fn new() -> Self {
        Self {
            hits_head: AtomicU64::new(0),
            hits_tail: AtomicU64::new(0),
            hits_resume: AtomicU64::new(0),
            redirects_issued: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            prewarms_started: AtomicU64::new(0),
            prewarms_high_started: AtomicU64::new(0),
            prewarms_low_started: AtomicU64::new(0),
            prewarms_succeeded: AtomicU64::new(0),
            prewarms_failed: AtomicU64::new(0),
            bytes_served_lan: AtomicU64::new(0),
        }
    }
}

pub static METRICS: FastStartMetrics = FastStartMetrics::new();

/// Maximum aggregate RAM held in the fast-start buffer cache (2 GB limit).
pub const FAST_START_MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Fast-start buffer cache, keyed by upstream stream URL.
///
/// Bounded strictly to 2 GB aggregate RAM via LRU weigher policy.
/// Immutable video buffers remain in RAM until evicted by the 2 GB LRU cap or explicitly invalidated
/// by a failed liveness check; `direct_cdn_url` and `created_at` are refreshed in-place on 8-byte liveness checks.
pub(crate) static FAST_START_CACHE: LazyLock<moka::sync::Cache<String, FastStartEntry>> =
    LazyLock::new(|| {
        moka::sync::Cache::builder()
            .max_capacity(FAST_START_MAX_BYTES)
            .weigher(|_k, v: &FastStartEntry| {
                let bytes = v.head_bytes.len()
                    + v.tail_bytes.as_ref().map_or(0, |b| b.len())
                    + v.resume_buffer.as_ref().map_or(0, |(_, b)| b.len())
                    + 1024;
                bytes.min(u32::MAX as usize) as u32
            })
            .build()
    });

/// Tracks in-flight pre-warms so requests for uncompleted buffers can stream chunks immediately.
pub(crate) static IN_FLIGHT_PREWARMS: LazyLock<dashmap::DashMap<String, Arc<InFlightHead>>> =
    LazyLock::new(dashmap::DashMap::new);

/// Tracks URLs currently queued or running in the low-priority background lane to prevent duplicate spawns.
static QUEUED_PREWARMS: LazyLock<dashmap::DashSet<String>> =
    LazyLock::new(dashmap::DashSet::new);

/// Keyed lock held ONLY while actively downloading a URL (after any background semaphore permit is acquired).
static PREWARM_LOCKS: LazyLock<crate::keyed_lock::KeyedLock<String>> =
    LazyLock::new(crate::keyed_lock::KeyedLock::new);

/// Entry tracking an active fast-start pre-warm notification handle and its active reference count.
pub struct PrewarmNotifyEntry {
    pub notify: Arc<tokio::sync::Notify>,
    pub refs: std::sync::atomic::AtomicUsize,
}

/// RAII guard that keeps an `ACTIVE_PREWARM_NOTIFY` entry registered until all pre-warm handles for a URL finish.
pub struct ActivePrewarmNotifyGuard {
    url: String,
    entry: Option<Arc<PrewarmNotifyEntry>>,
}

impl ActivePrewarmNotifyGuard {
    pub fn notify_waiters(&self) {
        if let Some(entry) = &self.entry {
            entry.notify.notify_waiters();
        }
    }
}

impl Drop for ActivePrewarmNotifyGuard {
    fn drop(&mut self) {
        if let Some(entry) = self.entry.take() {
            if entry.refs.fetch_sub(1, Ordering::SeqCst) == 1 {
                ACTIVE_PREWARM_NOTIFY
                    .remove_if(&self.url, |_, current| Arc::ptr_eq(current, &entry));
                entry.notify.notify_waiters();
            }
        }
    }
}

/// Tracks URLs actively executing an initial pre-warm head request so concurrent liveness checks can coalesce.
pub static ACTIVE_PREWARM_NOTIFY: LazyLock<dashmap::DashMap<String, Arc<PrewarmNotifyEntry>>> =
    LazyLock::new(dashmap::DashMap::new);

/// Synchronously registers an active pre-warm notification guard for `url`.
/// When the last `ActivePrewarmNotifyGuard` for `url` drops, the entry is removed and waiters are woken.
pub fn register_active_prewarm_notify(url: &str) -> ActivePrewarmNotifyGuard {
    if is_fast_start_cached(url) {
        return ActivePrewarmNotifyGuard {
            url: url.to_string(),
            entry: None,
        };
    }
    let entry = ACTIVE_PREWARM_NOTIFY
        .entry(url.to_string())
        .or_insert_with(|| {
            Arc::new(PrewarmNotifyEntry {
                notify: Arc::new(tokio::sync::Notify::new()),
                refs: std::sync::atomic::AtomicUsize::new(0),
            })
        })
        .clone();
    entry.refs.fetch_add(1, Ordering::SeqCst);
    ActivePrewarmNotifyGuard {
        url: url.to_string(),
        entry: Some(entry),
    }
}

/// Returns the `Notify` handle if a fast-start pre-warm is actively running for `url`.
pub fn get_active_prewarm_notify(url: &str) -> Option<Arc<tokio::sync::Notify>> {
    ACTIVE_PREWARM_NOTIFY.get(url).map(|n| n.value().notify.clone())
}

/// Returns whether the given URL currently has a pre-warmed entry in `FAST_START_CACHE` with valid head bytes.
pub fn is_fast_start_cached(url: &str) -> bool {
    FAST_START_CACHE.get(url).is_some_and(|e| !e.head_bytes.is_empty())
}

pub(crate) static FAST_START_DB: std::sync::OnceLock<sqlx::SqlitePool> = std::sync::OnceLock::new();

pub fn set_fast_start_db(db: sqlx::SqlitePool) {
    let _ = FAST_START_DB.set(db);
}

pub fn get_fast_start_db() -> Option<&'static sqlx::SqlitePool> {
    FAST_START_DB.get()
}

static UPGRADING_PROBES: LazyLock<dashmap::DashSet<String>> =
    LazyLock::new(dashmap::DashSet::new);

/// Bounded set of stream URLs already verified/upgraded with reusable probe data in SQLite,
/// avoiding redundant DB lookups on repeat cache hits in `refresh_cached_resume_and_head`.
static PROBED_UPGRADE_DONE: LazyLock<moka::sync::Cache<String, ()>> = LazyLock::new(|| {
    moka::sync::Cache::builder()
        .max_capacity(2048)
        .time_to_live(Duration::from_secs(6 * 3600))
        .build()
});

struct ProbeUpgradeGuard(String);
impl Drop for ProbeUpgradeGuard {
    fn drop(&mut self) {
        UPGRADING_PROBES.remove(&self.0);
    }
}

/// Global semaphore limiting concurrent low-priority background pre-warms to 2 simultaneous chunk downloads,
/// leaving bandwidth headroom for high-priority interactive playback requests.
pub static FAST_START_PREWARM_SEMAPHORE: LazyLock<tokio::sync::Semaphore> =
    LazyLock::new(|| tokio::sync::Semaphore::new(2));

/// Dedicated reqwest client for pre-warming (follows HTTP 302/307 redirects to CDN).
static PREWARM_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(120))
        .tcp_keepalive(Some(Duration::from_secs(15)))
        .pool_idle_timeout(Some(Duration::from_secs(90)))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .expect("failed to build fast-start prewarm client")
});

async fn refresh_cached_resume_and_head(
    item_id: Option<Uuid>,
    original_url: &str,
    request_headers: &HashMap<String, String>,
    is_4k_hint: Option<bool>,
    resume_position_secs: Option<f64>,
) -> bool {
    let Some(entry) = FAST_START_CACHE.get(original_url) else {
        return false;
    };

    // Attach or normalize item_id if previously missing or tagged with a child stream UUID
    if let Some(id) = item_id {
        if entry.item_id != Some(id) {
            let mut updated = entry.clone();
            updated.item_id = Some(id);
            FAST_START_CACHE.insert(original_url.to_string(), updated);
        }
    }

    if let Some(pos_secs) = resume_position_secs {
        if pos_secs > 10.0 {
            let is_4k = is_4k_stream(original_url, Some(entry.total_size), is_4k_hint);
            let target_resume_head = if is_4k {
                FAST_START_RESUME_HEAD_SIZE_4K
            } else {
                FAST_START_RESUME_HEAD_SIZE_1080P
            };

            // Expand cached head buffer to 64MB if previously cached with smaller head
            if is_4k && entry.head_bytes.len() < target_resume_head && entry.total_size > entry.head_bytes.len() as u64 {
                let head_lock_key = format!("head:{}", original_url);
                let _head_guard = PREWARM_LOCKS.lock(head_lock_key).await;
                if let Some(current_entry) = FAST_START_CACHE.get(original_url) {
                    if current_entry.head_bytes.len() < target_resume_head && current_entry.total_size > current_entry.head_bytes.len() as u64 {
                        let missing_start = current_entry.head_bytes.len();
                        let missing_end = (target_resume_head as u64 - 1).min(current_entry.total_size - 1);
                        let missing_range = format!("bytes={}-{}", missing_start, missing_end);
                        let mut head_req = PREWARM_CLIENT.get(&current_entry.direct_cdn_url);
                        for (k, v) in request_headers {
                            head_req = head_req.header(k.as_str(), v.as_str());
                        }
                        head_req = head_req.header(header::RANGE, missing_range.clone());
                        let head_resp = match head_req.send().await {
                            Ok(r) if r.status() == StatusCode::PARTIAL_CONTENT => Ok(r),
                            _ => {
                                let mut fallback_req = PREWARM_CLIENT.get(original_url);
                                for (k, v) in request_headers {
                                    fallback_req = fallback_req.header(k.as_str(), v.as_str());
                                }
                                fallback_req.header(header::RANGE, missing_range).send().await
                            }
                        };
                        if let Ok(resp) = head_resp {
                            if resp.status() == StatusCode::PARTIAL_CONTENT {
                                if let Ok(extra) = resp.bytes().await {
                                    if let Some(mut updated) = FAST_START_CACHE.get(original_url) {
                                        if updated.head_bytes.len() == missing_start {
                                            let mut combined = BytesMut::with_capacity(updated.head_bytes.len() + extra.len());
                                            combined.extend_from_slice(&updated.head_bytes);
                                            combined.extend_from_slice(&extra);
                                            updated.head_bytes = combined.freeze();
                                            FAST_START_CACHE.insert(original_url.to_string(), updated);
                                            info!(
                                                %original_url,
                                                new_head_len = FAST_START_CACHE.get(original_url).map(|e| e.head_bytes.len()).unwrap_or(0),
                                                "[FAST-START] Expanded cached head buffer to 64MB for 4K resume"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let entry = FAST_START_CACHE.get(original_url).unwrap_or_else(|| entry.clone());

            if let Some(ref tail) = entry.tail_bytes {
                if let Some(target_offset) = find_mkv_cue_offset(&entry.head_bytes, tail, pos_secs) {
                    let preroll_bytes = if is_4k {
                        FAST_START_RESUME_PREROLL_BYTES_4K as u64
                    } else {
                        FAST_START_RESUME_PREROLL_BYTES_1080P as u64
                    };
                    let resume_buffer_size = if is_4k {
                        FAST_START_RESUME_BUFFER_SIZE_4K
                    } else {
                        FAST_START_RESUME_BUFFER_SIZE_1080P
                    };
                    let min_forward_bytes =
                        (resume_buffer_size as u64).saturating_sub(preroll_bytes) / 2;
                    let min_end = (target_offset + min_forward_bytes).min(entry.total_size);
                    let already_covered = entry.resume_buffer.as_ref().map(|(start, bytes)| {
                        let end = *start + bytes.len() as u64;
                        let min_start = target_offset.saturating_sub(preroll_bytes / 2);
                        target_offset >= *start && end >= min_end && *start <= min_start
                    }).unwrap_or(false);

                    if !already_covered {
                        let resume_lock_key = format!("resume:{}", original_url);
                        let _guard = PREWARM_LOCKS.lock(resume_lock_key).await;
                        // Re-check after acquiring lock in case a concurrent task already refreshed it
                        if let Some(current_entry) = FAST_START_CACHE.get(original_url) {
                            let covered_now = current_entry.resume_buffer.as_ref().map(|(start, bytes)| {
                                let end = *start + bytes.len() as u64;
                                let min_start = target_offset.saturating_sub(preroll_bytes / 2);
                                target_offset >= *start && end >= min_end && *start <= min_start
                            }).unwrap_or(false);
                            if covered_now {
                                return true;
                            }
                        }

                        let current_entry = FAST_START_CACHE.get(original_url).unwrap_or(entry.clone());
                        let start_byte = target_offset.saturating_sub(preroll_bytes).max(current_entry.head_bytes.len() as u64);
                        let end_byte = (start_byte + resume_buffer_size as u64 - 1).min(current_entry.total_size - 1);
                        let resume_range = format!("bytes={}-{}", start_byte, end_byte);

                        let mut res_req = PREWARM_CLIENT.get(&current_entry.direct_cdn_url);
                        for (k, v) in request_headers {
                            res_req = res_req.header(k.as_str(), v.as_str());
                        }
                        res_req = res_req.header(header::RANGE, resume_range.clone());
                        let resp_res = match res_req.send().await {
                            Ok(r) if r.status() == StatusCode::PARTIAL_CONTENT => Ok(r),
                            other => {
                                warn!(%original_url, status = ?other.as_ref().map(|r| r.status()), "[FAST-START] Direct CDN resume fetch failed or expired, falling back to original_url");
                                let mut fallback_req = PREWARM_CLIENT.get(original_url);
                                for (k, v) in request_headers {
                                    fallback_req = fallback_req.header(k.as_str(), v.as_str());
                                }
                                fallback_req = fallback_req.header(header::RANGE, resume_range);
                                match fallback_req.send().await {
                                    Ok(r) if r.status() == StatusCode::PARTIAL_CONTENT => {
                                        let fresh_cdn = r.url().as_str().to_string();
                                        if let Some(mut updated) = FAST_START_CACHE.get(original_url) {
                                            updated.direct_cdn_url = fresh_cdn;
                                            FAST_START_CACHE.insert(original_url.to_string(), updated);
                                        }
                                        Ok(r)
                                    }
                                    err => err,
                                }
                            }
                        };
                        if let Ok(resp) = resp_res {
                            if resp.status() == StatusCode::PARTIAL_CONTENT {
                                if let Ok(b) = resp.bytes().await {
                                    let resume_size_bytes = b.len();
                                    if let Some(mut updated) = FAST_START_CACHE.get(original_url) {
                                        if let Some(id) = item_id {
                                            if updated.item_id != Some(id) {
                                                updated.item_id = Some(id);
                                            }
                                        }
                                        updated.resume_buffer = Some((start_byte, b));
                                        FAST_START_CACHE.insert(original_url.to_string(), updated);
                                        info!(
                                            %original_url,
                                            item = ?item_id,
                                            start_byte,
                                            end_byte,
                                            resume_size_bytes,
                                            "[FAST-START] Refreshed resume buffer into RAM for cached stream"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    debug!(
        %original_url,
        item = ?item_id,
        head_bytes = entry.head_bytes.len(),
        "[FAST-START] Stream already cached; skipping prewarm"
    );

    if let Some(db) = get_fast_start_db() {
        spawn_probe_upgrade(db.clone(), item_id, original_url.to_string());
    }

    true
}

/// Pre-warms the head buffer (32 MB or 8 MB resume), tail index (2 MB), and keyframe resume chunk (24/48 MB) for an upstream stream.
///
/// Runs asynchronously in the background. Follows upstream redirects (e.g. Stremthru -> Torbox CDN),
/// resolves the direct CDN edge URL, and caches the bytes in memory.
pub async fn prewarm_fast_start(
    item_id: Option<Uuid>,
    original_url: &str,
    request_headers: &HashMap<String, String>,
    priority: PrewarmPriority,
    head_size: Option<usize>,
    resume_position_secs: Option<f64>,
) -> bool {
    let is_4k_hint = head_size.map(|sz| sz >= FAST_START_4K_HEAD_SIZE);

    // If already pre-warmed and cached: check if resume buffer needs to be appended or refreshed
    if FAST_START_CACHE.get(original_url).is_some() {
        return refresh_cached_resume_and_head(
            item_id,
            original_url,
            request_headers,
            is_4k_hint,
            resume_position_secs,
        )
        .await;
    }

    // If FORCE_DIRECT_307 is enabled, do NOT download 32MB head or 2MB tail into RAM!
    // Only resolve the redirect chain to capture the direct Debrid CDN edge URL in ~50ms (0 bytes buffered in RAM).
    if is_force_direct_307() {
        if FAST_START_CACHE.get(original_url).is_some() {
            return true;
        }
        let _guard = PREWARM_LOCKS.lock(original_url.to_string()).await;
        if FAST_START_CACHE.get(original_url).is_some() {
            return true;
        }

        let mut head_req = PREWARM_CLIENT.get(original_url);
        for (k, v) in request_headers {
            head_req = head_req.header(k.as_str(), v.as_str());
        }
        head_req = head_req.header(header::RANGE, "bytes=0-0");
        let head_resp = match head_req.send().await {
            Ok(r) => r,
            Err(e) => {
                METRICS.prewarms_failed.fetch_add(1, Ordering::Relaxed);
                warn!(%original_url, error = %e, "[FAST-START] Direct-307 resolution request failed");
                if e.is_timeout() {
                    crate::stream::record_upstream_timeout("fast-start direct-307 timeout");
                }
                return false;
            }
        };

        let status = head_resp.status();
        if status == StatusCode::TOO_MANY_REQUESTS
            || status == StatusCode::GATEWAY_TIMEOUT
            || status == StatusCode::SERVICE_UNAVAILABLE
        {
            let retry_after = head_resp
                .headers()
                .get(header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .map(std::time::Duration::from_secs);
            crate::stream::trip_preemptive_circuit_breaker(retry_after, "fast-start direct-307 rate-limit/gateway error");
            crate::stream::STREAM_LIVENESS_CACHE.insert(
                original_url.to_string(),
                crate::stream::LivenessEntry::new_rate_limited_or_timeout(),
            );
            return false;
        }

        let direct_cdn_url = head_resp.url().as_str().to_string();
        let total_size = head_resp
            .headers()
            .get(header::CONTENT_RANGE)
            .and_then(|v| v.to_str().ok())
            .and_then(|cr| cr.split('/').last())
            .and_then(|s| s.parse::<u64>().ok())
            .or_else(|| head_resp.content_length())
            .unwrap_or(0);

        let is_tiny = total_size > 0 && total_size < 2 * 1024 * 1024;
        if crate::stream::is_error_video_url(&direct_cdn_url) || crate::stream::is_error_video_url(original_url) || is_tiny {
            let is_rate_limit = crate::stream::is_upstream_rate_limit_url(&direct_cdn_url)
                || crate::stream::is_upstream_rate_limit_url(original_url);
            warn!(%original_url, direct_url = %direct_cdn_url, total_size, is_rate_limit, "[FAST-START] Upstream resolved to error video URL or tiny payload; marking dead");
            if is_rate_limit {
                crate::stream::trip_preemptive_circuit_breaker(None, "fast-start direct-307 rate-limit video URL");
                crate::stream::STREAM_LIVENESS_CACHE.insert(
                    original_url.to_string(),
                    crate::stream::LivenessEntry::new_rate_limited_or_timeout(),
                );
            } else {
                crate::stream::STREAM_LIVENESS_CACHE.insert(
                    original_url.to_string(),
                    crate::stream::LivenessEntry::new(false),
                );
            }
            return false;
        }

        crate::stream::record_upstream_success();

        crate::stream::STREAM_LIVENESS_CACHE.insert(
            original_url.to_string(),
            crate::stream::LivenessEntry::new(true),
        );
        let content_type = detect_content_type(original_url).to_string();

        let entry = FastStartEntry {
            item_id,
            original_url: original_url.to_string(),
            direct_cdn_url: direct_cdn_url.clone(),
            total_size,
            head_bytes: Bytes::new(),
            tail_bytes: None,
            resume_buffer: None,
            content_type,
            created_at: Instant::now(),
        };
        FAST_START_CACHE.insert(original_url.to_string(), entry);
        info!(
            %original_url,
            item = ?item_id,
            direct_cdn_host = head_resp_host(&direct_cdn_url),
            "[FAST-START] Resolved direct Debrid CDN URL (0 bytes RAM buffered, FORCE_DIRECT_307=true)"
        );
        return true;
    }

    struct QueuedPrewarmGuard(Option<String>);
    impl Drop for QueuedPrewarmGuard {
        fn drop(&mut self) {
            if let Some(ref url) = self.0 {
                QUEUED_PREWARMS.remove(url);
            }
        }
    }

    // Low-priority tasks deduplicate via QUEUED_PREWARMS and wait for a background semaphore permit
    // BEFORE acquiring PREWARM_LOCKS so they never block foreground playback requests or High-priority prewarms.
    let (_queued_guard, _permit) = match priority {
        PrewarmPriority::Low => {
            if crate::stream::is_preemptive_probe_circuit_broken() {
                debug!(
                    %original_url,
                    item = ?item_id,
                    "[FAST-START] Low-priority pre-warm skipped: circuit breaker active due to upstream rate limit/outage"
                );
                return false;
            }
            if !QUEUED_PREWARMS.insert(original_url.to_string()) {
                debug!(
                    %original_url,
                    item = ?item_id,
                    "[FAST-START] Low-priority pre-warm already queued or in-flight; skipping duplicate"
                );
                return false;
            }
            let q_guard = QueuedPrewarmGuard(Some(original_url.to_string()));
            let permit = match tokio::time::timeout(
                Duration::from_secs(60),
                FAST_START_PREWARM_SEMAPHORE.acquire(),
            )
            .await
            {
                Ok(Ok(p)) => p,
                _ => {
                    warn!(%original_url, "[FAST-START] Timed out waiting for low-priority pre-warm semaphore permit, skipping");
                    return false;
                }
            };
            if crate::stream::is_preemptive_probe_circuit_broken() {
                debug!(
                    %original_url,
                    item = ?item_id,
                    "[FAST-START] Low-priority pre-warm aborted after acquiring permit: circuit breaker active"
                );
                return false;
            }
            (q_guard, Some(permit))
        }
        PrewarmPriority::High => (QueuedPrewarmGuard(None), None),
    };

    // Re-check cache before acquiring active lock (in case a High-priority task finished while Low was queued)
    if FAST_START_CACHE.get(original_url).is_some() {
        return refresh_cached_resume_and_head(
            item_id,
            original_url,
            request_headers,
            is_4k_hint,
            resume_position_secs,
        )
        .await;
    }

    // Coalesce concurrent active downloads for the same stream URL
    let _guard = PREWARM_LOCKS.lock(original_url.to_string()).await;

    // Double check cache after acquiring active lock
    if FAST_START_CACHE.get(original_url).is_some() {
        return refresh_cached_resume_and_head(
            item_id,
            original_url,
            request_headers,
            is_4k_hint,
            resume_position_secs,
        )
        .await;
    }

    let notify_guard = register_active_prewarm_notify(original_url);

    METRICS.prewarms_started.fetch_add(1, Ordering::Relaxed);
    match priority {
        PrewarmPriority::High => METRICS.prewarms_high_started.fetch_add(1, Ordering::Relaxed),
        PrewarmPriority::Low => METRICS.prewarms_low_started.fetch_add(1, Ordering::Relaxed),
    };
    let start_time = Instant::now();

    let is_resuming = resume_position_secs.map(|p| p > 10.0).unwrap_or(false);
    let target_head_size = if is_resuming {
        calculate_fast_start_resume_head_size(original_url, None, is_4k_hint)
    } else {
        head_size.unwrap_or_else(|| calculate_fast_start_head_size(original_url, None, is_4k_hint))
    };

    info!(
        %original_url,
        item = ?item_id,
        ?priority,
        target_size_mb = target_head_size / (1024 * 1024),
        is_resuming,
        ?resume_position_secs,
        "[FAST-START] Starting pre-warm download"
    );

    // Step 1: Request the first N MB (32 MB for standard / 4K)
    let head_range = format!("bytes=0-{}", target_head_size - 1);
    let build_head_req = || {
        let mut req = PREWARM_CLIENT.get(original_url);
        for (k, v) in request_headers {
            req = req.header(k.as_str(), v.as_str());
        }
        req.header(header::RANGE, head_range.clone())
    };

    let prewarm_timeout_secs = std::env::var("REMUX_LIVENESS_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(12);
    let mut head_resp_result = match tokio::time::timeout(Duration::from_secs(prewarm_timeout_secs), build_head_req().send()).await {
        Ok(res) => res,
        Err(_) => {
            METRICS.prewarms_failed.fetch_add(1, Ordering::Relaxed);
            warn!(%original_url, timeout_secs = prewarm_timeout_secs, "[FAST-START] Pre-warm head request timed out");
            crate::stream::record_upstream_timeout("fast-start head request timed out");
            crate::stream::STREAM_LIVENESS_CACHE.insert(
                original_url.to_string(),
                crate::stream::LivenessEntry::new_rate_limited_or_timeout(),
            );
            notify_guard.notify_waiters();
            return false;
        }
    };

    // Retry once on ephemeral 502 Bad Gateway or transient connection error
    let should_retry = match &head_resp_result {
        Ok(resp) => resp.status() == StatusCode::BAD_GATEWAY,
        Err(e) => !e.is_timeout(),
    };
    if should_retry {
        warn!(%original_url, "[FAST-START] Pre-warm head request encountered 502/network glitch; retrying once after 500ms");
        tokio::time::sleep(Duration::from_millis(500)).await;
        if let Ok(res) = tokio::time::timeout(Duration::from_secs(prewarm_timeout_secs), build_head_req().send()).await {
            head_resp_result = res;
        }
    }

    let head_resp = match head_resp_result {
        Ok(r) => r,
        Err(e) => {
            METRICS.prewarms_failed.fetch_add(1, Ordering::Relaxed);
            warn!(%original_url, error = %e, "[FAST-START] Pre-warm head request failed");
            if e.is_timeout() {
                crate::stream::record_upstream_timeout("fast-start head request timeout");
                crate::stream::STREAM_LIVENESS_CACHE.insert(
                    original_url.to_string(),
                    crate::stream::LivenessEntry::new_rate_limited_or_timeout(),
                );
            } else {
                crate::stream::STREAM_LIVENESS_CACHE.insert(
                    original_url.to_string(),
                    crate::stream::LivenessEntry::new(false),
                );
            }
            notify_guard.notify_waiters();
            return false;
        }
    };

    let status = head_resp.status();
    let is_partial = status == StatusCode::PARTIAL_CONTENT;
    let is_ok = status == StatusCode::OK;

    if status == StatusCode::TOO_MANY_REQUESTS
        || status == StatusCode::GATEWAY_TIMEOUT
        || status == StatusCode::SERVICE_UNAVAILABLE
        || status == StatusCode::BAD_GATEWAY
    {
        let retry_after = head_resp
            .headers()
            .get(header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .map(std::time::Duration::from_secs);
        crate::stream::trip_preemptive_circuit_breaker(retry_after, "fast-start pre-warm rate-limit/gateway error");
        crate::stream::STREAM_LIVENESS_CACHE.insert(
            original_url.to_string(),
            crate::stream::LivenessEntry::new_rate_limited_or_timeout(),
        );
        notify_guard.notify_waiters();
        return false;
    }

    if !is_partial && !is_ok {
        METRICS.prewarms_failed.fetch_add(1, Ordering::Relaxed);
        warn!(%original_url, ?status, "[FAST-START] Pre-warm head request returned non-success status");
        crate::stream::STREAM_LIVENESS_CACHE.insert(
            original_url.to_string(),
            crate::stream::LivenessEntry::new(false),
        );
        notify_guard.notify_waiters();
        return false;
    }

    // Capture the direct CDN edge URL after Stremthru redirect
    let direct_cdn_url = head_resp.url().as_str().to_string();

    // Parse total file size from Content-Range header: bytes 0-16777215/4694973563
    let total_size = head_resp
        .headers()
        .get(header::CONTENT_RANGE)
        .and_then(|v| v.to_str().ok())
        .and_then(|cr| cr.split('/').last())
        .and_then(|s| s.parse::<u64>().ok())
        .or_else(|| if is_ok { head_resp.content_length() } else { None });

    let total_size = match total_size {
        Some(s) => s,
        None => {
            METRICS.prewarms_failed.fetch_add(1, Ordering::Relaxed);
            warn!(%original_url, "[FAST-START] Could not determine total file size for prewarm");
            return false;
        }
    };

    let is_tiny_video = total_size < 2 * 1024 * 1024;
    if crate::stream::is_error_video_url(&direct_cdn_url) || crate::stream::is_error_video_url(original_url) || is_tiny_video {
        METRICS.prewarms_failed.fetch_add(1, Ordering::Relaxed);
        let is_rate_limit = crate::stream::is_upstream_rate_limit_url(&direct_cdn_url)
            || crate::stream::is_upstream_rate_limit_url(original_url);
        warn!(
            %original_url,
            direct_url = %direct_cdn_url,
            total_size,
            is_rate_limit,
            "[FAST-START] Upstream resolved to error placeholder video URL or tiny payload; marking dead"
        );
        if is_rate_limit {
            crate::stream::trip_preemptive_circuit_breaker(None, "fast-start upstream rate-limit video detected");
            crate::stream::STREAM_LIVENESS_CACHE.insert(
                original_url.to_string(),
                crate::stream::LivenessEntry::new_rate_limited_or_timeout(),
            );
        } else {
            crate::stream::STREAM_LIVENESS_CACHE.insert(
                original_url.to_string(),
                crate::stream::LivenessEntry::new(false),
            );
        }
        crate::fast_start::FAST_START_CACHE.invalidate(original_url);
        notify_guard.notify_waiters();
        return false;
    }

    crate::stream::record_upstream_success();

    // Stream responded with 200/206 and valid size: populate STREAM_LIVENESS_CACHE immediately so liveness checks are instant
    crate::stream::STREAM_LIVENESS_CACHE.insert(
        original_url.to_string(),
        crate::stream::LivenessEntry::new(true),
    );
    notify_guard.notify_waiters();

    // Safety guard against OOM: if upstream returned 200 OK (ignored Range), only allow if file is small (<= target_head_size + 1024)
    if is_ok && total_size > (target_head_size + 1024) as u64 {
        METRICS.prewarms_failed.fetch_add(1, Ordering::Relaxed);
        warn!(
            %original_url,
            total_size,
            "[FAST-START] Upstream returned 200 OK instead of 206 Partial Content for large file; aborting to prevent OOM"
        );
        return false;
    }

    // Detect Content-Type from upstream response or file extension.
    // Exclude generic application/octet-stream so it falls back to explicit video MIME detection.
    let content_type = head_resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .filter(|ct| {
            (ct.starts_with("video/") || ct.starts_with("application/"))
                && !ct.starts_with("application/octet-stream")
        })
        .map(|ct| ct.to_string())
        .unwrap_or_else(|| detect_content_type(original_url).to_string());

    // Register in-flight pre-warm entry so concurrent stream requests can stream chunks immediately
    let in_flight = Arc::new(InFlightHead {
        item_id,
        original_url: original_url.to_string(),
        direct_cdn_url: direct_cdn_url.clone(),
        total_size,
        target_head_size,
        content_type: content_type.clone(),
        buffer: RwLock::new(BytesMut::with_capacity(target_head_size)),
        tail_bytes: RwLock::new(None),
        notify: tokio::sync::Notify::new(),
        tail_notify: tokio::sync::Notify::new(),
        is_done: AtomicBool::new(false),
        has_error: AtomicBool::new(false),
    });
    IN_FLIGHT_PREWARMS.insert(original_url.to_string(), in_flight.clone());
    // Wake any playback requests waiting for this active pre-warm to register its in-flight buffer
    notify_guard.notify_waiters();

    struct InFlightGuard<'a>(&'a str);
    impl<'a> Drop for InFlightGuard<'a> {
        fn drop(&mut self) {
            IN_FLIGHT_PREWARMS.remove(self.0);
        }
    }
    let _in_flight_guard = InFlightGuard(original_url);

    // Step 2: Stream head body in chunks AND fetch tail cues in parallel!
    let in_flight_tail = in_flight.clone();
    let direct_cdn_url_tail = direct_cdn_url.clone();
    let tail_fut = async move {
        let res = if total_size > FAST_START_TAIL_SIZE as u64 {
            let tail_start = total_size - FAST_START_TAIL_SIZE as u64;
            let tail_range = format!("bytes={}-{}", tail_start, total_size - 1);
            let mut tail_req = PREWARM_CLIENT.get(&direct_cdn_url_tail);
            for (k, v) in request_headers {
                tail_req = tail_req.header(k.as_str(), v.as_str());
            }
            tail_req = tail_req.header(header::RANGE, tail_range);

            match tail_req.send().await {
                // Tail fetch MUST be HTTP 206 Partial Content. If server returned 200 OK, it ignored Range!
                Ok(r) if r.status() == StatusCode::PARTIAL_CONTENT => r.bytes().await.ok(),
                Ok(r) => {
                    warn!(%original_url, status = ?r.status(), "[FAST-START] Tail fetch returned non-206 status (Range ignored or error)");
                    None
                }
                Err(e) => {
                    warn!(%original_url, error = %e, "[FAST-START] Tail fetch request error");
                    None
                }
            }
        } else {
            None
        };
        {
            let mut t = in_flight_tail.tail_bytes.write().unwrap();
            *t = res.clone();
        }
        in_flight_tail.tail_notify.notify_waiters();
        res
    };

    let in_flight_head = in_flight.clone();
    let head_fut = async move {
        let mut stream = head_resp.bytes_stream();
        while let Some(chunk_res) = stream.next().await {
            match chunk_res {
                Ok(chunk) => {
                    {
                        let mut buf = in_flight_head.buffer.write().unwrap();
                        buf.extend_from_slice(&chunk);
                    }
                    in_flight_head.notify.notify_waiters();
                }
                Err(e) => {
                    in_flight_head.has_error.store(true, Ordering::Relaxed);
                    in_flight_head.notify.notify_waiters();
                    return Err(e);
                }
            }
        }
        in_flight_head.is_done.store(true, Ordering::Relaxed);
        in_flight_head.notify.notify_waiters();
        let final_bytes = {
            let buf = in_flight_head.buffer.read().unwrap();
            Bytes::copy_from_slice(&buf)
        };
        Ok(final_bytes)
    };

    let (head_res, tail_bytes) = tokio::join!(head_fut, tail_fut);

    let head_bytes = match head_res {
        Ok(b) => b,
        Err(e) => {
            METRICS.prewarms_failed.fetch_add(1, Ordering::Relaxed);
            warn!(%original_url, error = %e, "[FAST-START] Failed reading head response body (falling back to direct CDN streaming)");
            return false;
        }
    };

    // Step 3: If resuming, extract video cue offset and pre-warm resume buffer with backward headroom
    let resume_buffer = if let (Some(pos_secs), Some(tb)) = (resume_position_secs, &tail_bytes) {
        if pos_secs > 10.0 {
            if let Some(target_offset) = find_mkv_cue_offset(&head_bytes, tb, pos_secs) {
                let is_4k = is_4k_stream(original_url, Some(total_size), is_4k_hint);
                let preroll_bytes = if is_4k {
                    FAST_START_RESUME_PREROLL_BYTES_4K as u64
                } else {
                    FAST_START_RESUME_PREROLL_BYTES_1080P as u64
                };
                let target_resume_buffer_size = if is_4k {
                    FAST_START_RESUME_BUFFER_SIZE_4K
                } else {
                    FAST_START_RESUME_BUFFER_SIZE_1080P
                };
                let start_byte = target_offset.saturating_sub(preroll_bytes).max(head_bytes.len() as u64);
                let end_byte = (start_byte + target_resume_buffer_size as u64 - 1).min(total_size - 1);
                let resume_range = format!("bytes={}-{}", start_byte, end_byte);
                let mut resume_res: Option<(u64, Bytes)> = None;
                for attempt in 1..=2 {
                    let mut res_req = PREWARM_CLIENT.get(&direct_cdn_url);
                    for (k, v) in request_headers {
                        res_req = res_req.header(k.as_str(), v.as_str());
                    }
                    res_req = res_req.header(header::RANGE, resume_range.clone());

                    match res_req.send().await {
                        Ok(r) if r.status() == StatusCode::PARTIAL_CONTENT => {
                            match r.bytes().await {
                                Ok(b) => {
                                    info!(
                                        %original_url,
                                        start_byte,
                                        end_byte,
                                        resume_size_bytes = b.len(),
                                        attempt,
                                        "[FAST-START] Successfully pre-warmed resume chunk into RAM"
                                    );
                                    resume_res = Some((start_byte, b));
                                    break;
                                }
                                Err(e) => {
                                    if attempt < 2 {
                                        tracing::debug!(%original_url, error = %e, "[FAST-START] Resume chunk read failed, retrying once...");
                                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                                        continue;
                                    }
                                    warn!(%original_url, error = %e, "[FAST-START] Failed reading resume chunk body");
                                }
                            }
                        }
                        Ok(r) => {
                            warn!(%original_url, status = ?r.status(), "[FAST-START] Resume fetch returned non-206 status");
                            break;
                        }
                        Err(e) => {
                            if attempt < 2 {
                                tracing::debug!(%original_url, error = %e, "[FAST-START] Resume fetch request error, retrying once...");
                                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                                continue;
                            }
                            warn!(%original_url, error = %e, "[FAST-START] Resume fetch request error");
                        }
                    }
                }
                resume_res
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    let entry = FastStartEntry {
        item_id,
        original_url: original_url.to_string(),
        direct_cdn_url: direct_cdn_url.clone(),
        total_size,
        head_bytes: head_bytes.clone(),
        tail_bytes: tail_bytes.clone(),
        resume_buffer: resume_buffer.clone(),
        content_type,
        created_at: Instant::now(),
    };

    FAST_START_CACHE.insert(original_url.to_string(), entry);

    METRICS.prewarms_succeeded.fetch_add(1, Ordering::Relaxed);
    let elapsed = start_time.elapsed();

    info!(
        %original_url,
        item = ?item_id,
        direct_cdn_host = head_resp_host(&direct_cdn_url),
        head_size_bytes = head_bytes.len(),
        tail_size_bytes = tail_bytes.as_ref().map(|b| b.len()).unwrap_or(0),
        resume_size_bytes = resume_buffer.as_ref().map(|(_, b)| b.len()).unwrap_or(0),
        total_size_bytes = total_size,
        ?elapsed,
        "[FAST-START] Successfully pre-warmed stream into RAM"
    );

    if let Some(db) = get_fast_start_db() {
        spawn_probe_upgrade(db.clone(), item_id, original_url.to_string());
    }

    true
}

fn head_resp_host(url: &str) -> &str {
    url.split("://")
        .nth(1)
        .and_then(|s| s.split('/').next())
        .unwrap_or(url)
}

static LOCAL_PROBE_PORT: tokio::sync::OnceCell<u16> = tokio::sync::OnceCell::const_new();

async fn ensure_local_probe_server() -> Option<u16> {
    LOCAL_PROBE_PORT
        .get_or_try_init(|| async {
            let app = axum::Router::new().route(
                "/fast-start-probe",
                axum::routing::get(
                    |axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
                     headers: HeaderMap| async move {
                        let Some(url) = params.get("url") else {
                            return StatusCode::BAD_REQUEST.into_response();
                        };
                        if let Some(resp) =
                            handle_playback_request(Uuid::nil(), url, &headers, None).await
                        {
                            resp
                        } else {
                            Redirect::temporary(url).into_response()
                        }
                    },
                ),
            );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
            let port = listener.local_addr()?.port();
            tokio::spawn(async move {
                if let Err(e) = axum::serve(listener, app).await {
                    tracing::error!(error = %e, "[FAST-START] Local RAM probe HTTP server terminated unexpectedly");
                }
            });
            Ok::<u16, std::io::Error>(port)
        })
        .await
        .ok()
        .copied()
}

/// Returns the localhost probe URL (`http://127.0.0.1:{port}/fast-start-probe?url=...`) if the
/// local probe server is listening.
pub async fn format_local_probe_url(url: &str) -> Option<String> {
    let port = ensure_local_probe_server().await?;
    let encoded: String = url::form_urlencoded::byte_serialize(url.as_bytes()).collect();
    info!(
        %url,
        local_port = port,
        "[FAST-START] Routing ffprobe through local RAM buffer"
    );
    Some(format!("http://127.0.0.1:{port}/fast-start-probe?url={encoded}"))
}

/// Ensures the given HTTP URL is pre-warmed in `FAST_START_CACHE` and returns a localhost
/// Fast-Start URL (`http://127.0.0.1:{port}/fast-start-probe?url=...`) so `ffprobe` can read
/// the 16 MB head and 256 KB tail cues directly from RAM in ~15ms instead of making multiple
/// round-trips through upstream resolvers. Falls back to the original URL if uncached.
pub async fn resolve_fast_start_probe_url(url: &str) -> String {
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return url.to_string();
    }
    if url.starts_with("http://127.0.0.1:") || url.starts_with("http://localhost:") {
        return url.to_string();
    }
    if !prewarm_fast_start(None, url, &HashMap::new(), PrewarmPriority::High, None, None).await {
        return url.to_string();
    }
    format_local_probe_url(url).await.unwrap_or_else(|| url.to_string())
}

/// Spawns a background task to upgrade candidate probe data from RAM buffer if unprobed.
pub fn spawn_probe_upgrade(
    db: sqlx::SqlitePool,
    item_id: Option<Uuid>,
    original_url: String,
) {
    let fut: std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> = Box::pin(async move {
        upgrade_probe_data_from_fast_start(&db, item_id, &original_url).await;
    });
    tokio::spawn(fut);
}

/// If a candidate stream in SQLite currently has missing or filename-guessed probe data,
/// probes it via the localhost Fast-Start RAM buffer (~15ms) and updates SQLite with real track metadata.
pub async fn upgrade_probe_data_from_fast_start(
    db: &sqlx::SqlitePool,
    item_id: Option<Uuid>,
    original_url: &str,
) {
    if PROBED_UPGRADE_DONE.contains_key(original_url) {
        return;
    }
    if !UPGRADING_PROBES.insert(original_url.to_string()) {
        return;
    }
    let _guard = ProbeUpgradeGuard(original_url.to_string());

    // 1. Look up stream in DB (scoped to kind = 'stream' so partial index idx_media_stream_http_url applies)
    let stream_opt = if let Some(id) = item_id.filter(|id| !id.is_nil()) {
        sqlx::query_as::<_, crate::db::Media>(
            "SELECT * FROM media WHERE kind = 'stream' AND (parent_id = ?1 OR id = ?1) AND json_extract(stream_info, '$.descriptor.Http.url') = ?2 LIMIT 1",
        )
        .bind(id)
        .bind(original_url)
        .fetch_optional(db)
        .await
        .ok()
        .flatten()
    } else {
        None
    };

    let stream = match stream_opt {
        Some(s) => s,
        None => {
            match sqlx::query_as::<_, crate::db::Media>(
                "SELECT * FROM media WHERE kind = 'stream' AND json_extract(stream_info, '$.descriptor.Http.url') = ?1 LIMIT 1",
            )
            .bind(original_url)
            .fetch_optional(db)
            .await
            {
                Ok(Some(s)) => s,
                _ => return,
            }
        }
    };

    // 2. Check if stream already has reusable probe data
    if let Some(cached) = &stream.probe_data {
        if crate::playback::probe::is_reusable_probe_cache(cached) {
            PROBED_UPGRADE_DONE.insert(original_url.to_string(), ());
            return;
        }
    }

    // 3. Resolve localhost probe URL (which serves head from FAST_START_CACHE in RAM)
    let Some(probe_url) = format_local_probe_url(original_url).await else {
        return;
    };
    let stream_id = stream.id;

    // 4. Run ffprobe against localhost in blocking thread pool with 3s timeout
    let probe_res = tokio::task::spawn_blocking(move || {
        crate::playback::probe::probe_media_with(
            &probe_url,
            None,
            Some(Duration::from_millis(3000)),
        )
    })
    .await;

    match probe_res {
        Ok(Ok((mut probed_info, _))) => {
            probed_info.id = stream_id;
            probed_info.e_tag = stream_id;
            crate::playback::probe::apply_video_bitrate_fallback(
                &mut probed_info.media_streams,
                probed_info.bitrate,
            );
            if let Err(e) = crate::db::Media::save_probe_data(db, &stream_id, &probed_info).await {
                warn!(%stream_id, error = %e, "[FAST-START] Failed to save upgraded probe data to DB");
            } else {
                PROBED_UPGRADE_DONE.insert(original_url.to_string(), ());
                info!(
                    %stream_id,
                    streams = probed_info.media_streams.len(),
                    "[FAST-START] Asynchronously upgraded candidate probe data from RAM buffer"
                );
            }
        }
        Ok(Err(e)) => {
            warn!(%stream_id, error = %e, "[FAST-START] Background RAM probe failed or incomplete");
        }
        Err(e) => {
            warn!(%stream_id, error = %e, "[FAST-START] Background RAM probe task join error");
        }
    }
}

/// Returns true when the `/stream` request comes from a web browser HTML5 `<video>` element
/// (e.g. Jellyfin Web). Chromium's `MultiBufferDataSource` enforces single-origin verification
/// across byte-range requests: if `bytes=0-` is served as `206` from Remux (`Origin A`) and a
/// subsequent seek/range returns `307` to the Debrid CDN (`Origin B`), the browser aborts playback.
/// Native clients (`Infuse`, `Lavf`, `VLC`, `mpv`) have no browser CORS/single-origin restriction.
fn is_browser_video_request(headers: &HeaderMap) -> bool {
    if headers.contains_key("sec-fetch-dest") || headers.contains_key("sec-fetch-mode") {
        return true;
    }
    headers
        .get(header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ua| ua.starts_with("Mozilla/"))
}

/// Serves chunks in real-time from an actively downloading in-flight pre-warm buffer.
async fn serve_from_in_flight(
    item_id: Uuid,
    in_flight: &Arc<InFlightHead>,
    headers: &HeaderMap,
) -> Option<Response> {
    let range_header = headers.get(header::RANGE).and_then(|v| v.to_str().ok());

    match range_header {
        // Case 1: Head sub-range or open-ended request (starts inside target_head_size)
        Some(rng) if is_head_range(rng, in_flight.target_head_size, in_flight.total_size) => {
            METRICS.hits_head.fetch_add(1, Ordering::Relaxed);
            let (req_start, req_end) = parse_range_bounds(rng, in_flight.total_size)
                .unwrap_or((0, in_flight.total_size.saturating_sub(1)));

            let max_chunk_end = req_start + (FAST_START_HEAD_SERVE_CHUNK_SIZE as u64).saturating_sub(1);
            let max_head_end = (in_flight.target_head_size as u64).saturating_sub(1);
            let end_offset = max_head_end
                .min(max_chunk_end)
                .min(req_end)
                .min(in_flight.total_size - 1);
            let content_len = end_offset.saturating_sub(req_start) + 1;

            info!(
                %item_id,
                req_start,
                end_offset,
                content_len,
                total_size = in_flight.total_size,
                "[FAST-START] Buffer hit (head): serving {} bytes locally over LAN (in-flight stream)",
                content_len
            );

            let in_flight_clone = in_flight.clone();
            let body_stream = async_stream::stream! {
                let mut cursor = req_start as usize;
                let target_end = (end_offset as usize) + 1;

                while cursor < target_end {
                    let chunk = {
                        let buf = in_flight_clone.buffer.read().unwrap();
                        let available_end = buf.len().min(target_end);
                        if available_end > cursor {
                            let slice = Bytes::copy_from_slice(&buf[cursor..available_end]);
                            cursor = available_end;
                            Some(slice)
                        } else {
                            None
                        }
                    };

                    if let Some(slice) = chunk {
                        let slice_len = slice.len();
                        METRICS.bytes_served_lan.fetch_add(slice_len as u64, Ordering::Relaxed);
                        yield Ok::<Bytes, std::io::Error>(slice);
                    } else {
                        // Register notified BEFORE checking is_done / has_error to prevent missing wakeups
                        let notified = in_flight_clone.notify.notified();
                        tokio::pin!(notified);
                        if in_flight_clone.is_done.load(Ordering::Relaxed) {
                            break;
                        }
                        if in_flight_clone.has_error.load(Ordering::Relaxed) {
                            yield Err::<Bytes, std::io::Error>(std::io::Error::new(
                                std::io::ErrorKind::ConnectionReset,
                                "in-flight stream error",
                            ));
                            break;
                        }
                        if tokio::time::timeout(
                            Duration::from_secs(5),
                            &mut notified,
                        ).await.is_err() {
                            tracing::warn!(
                                %item_id,
                                cursor,
                                target_end,
                                "[FAST-START] Upstream download wait timeout (5s); underrun detected"
                            );
                        }
                    }
                }
            };

            let resp = Response::builder()
                .status(StatusCode::PARTIAL_CONTENT)
                .header(header::CONTENT_TYPE, &in_flight.content_type)
                .header(
                    header::CONTENT_RANGE,
                    format!("bytes {}-{}/{}", req_start, end_offset, in_flight.total_size),
                )
                .header(header::CONTENT_LENGTH, content_len.to_string())
                .header(header::ACCEPT_RANGES, "bytes")
                .header("X-Remux-FastStart", "hit-head-inflight")
                .body(Body::from_stream(body_stream))
                .unwrap();

            Some(resp)
        }

        // Case 2: No range header specified (treat as head probe)
        None => {
            METRICS.hits_head.fetch_add(1, Ordering::Relaxed);
            let target_head_size = in_flight.target_head_size.min(FAST_START_HEAD_SERVE_CHUNK_SIZE);
            let end_offset = (target_head_size as u64).min(in_flight.total_size).saturating_sub(1);
            let content_len = end_offset + 1;

            info!(
                %item_id,
                content_len,
                total_size = in_flight.total_size,
                "[FAST-START] Buffer hit (no range): serving {} bytes locally over LAN (in-flight stream)",
                content_len
            );

            let in_flight_clone = in_flight.clone();
            let body_stream = async_stream::stream! {
                let mut cursor = 0usize;
                let target_end = (end_offset as usize) + 1;

                while cursor < target_end {
                    let chunk = {
                        let buf = in_flight_clone.buffer.read().unwrap();
                        let available_end = buf.len().min(target_end);
                        if available_end > cursor {
                            let slice = Bytes::copy_from_slice(&buf[cursor..available_end]);
                            cursor = available_end;
                            Some(slice)
                        } else {
                            None
                        }
                    };

                    if let Some(slice) = chunk {
                        let slice_len = slice.len();
                        METRICS.bytes_served_lan.fetch_add(slice_len as u64, Ordering::Relaxed);
                        yield Ok::<Bytes, std::io::Error>(slice);
                    } else {
                        if in_flight_clone.is_done.load(Ordering::Relaxed) {
                            break;
                        }
                        if in_flight_clone.has_error.load(Ordering::Relaxed) {
                            yield Err::<Bytes, std::io::Error>(std::io::Error::new(
                                std::io::ErrorKind::ConnectionReset,
                                "in-flight stream error",
                            ));
                            break;
                        }
                        let _ = tokio::time::timeout(
                            Duration::from_secs(5),
                            in_flight_clone.notify.notified(),
                        ).await;
                    }
                }
            };

            let resp = Response::builder()
                .status(StatusCode::PARTIAL_CONTENT)
                .header(header::CONTENT_TYPE, &in_flight.content_type)
                .header(
                    header::CONTENT_RANGE,
                    format!("bytes 0-{}/{}", end_offset, in_flight.total_size),
                )
                .header(header::CONTENT_LENGTH, content_len.to_string())
                .header(header::ACCEPT_RANGES, "bytes")
                .header("X-Remux-FastStart", "hit-head-inflight")
                .body(Body::from_stream(body_stream))
                .unwrap();

            Some(resp)
        }

        // Case 3: Tail / MKV Cues index probe
        Some(rng) if is_tail_range(rng, in_flight.total_size) => {
            let notified = in_flight.tail_notify.notified();
            tokio::pin!(notified);
            let tail_opt = in_flight.tail_bytes.read().unwrap().clone();
            let tail_bytes = if let Some(t) = tail_opt {
                Some(t)
            } else {
                let _ = tokio::time::timeout(
                    Duration::from_millis(3500),
                    &mut notified,
                ).await;
                in_flight.tail_bytes.read().unwrap().clone()
            };

            if let Some(tail) = tail_bytes {
                let tail_len = tail.len() as u64;
                let tail_start = in_flight.total_size.saturating_sub(tail_len);
                if let Some((req_start, req_end)) = parse_range_bounds(rng, in_flight.total_size) {
                    if req_start >= tail_start && req_start < in_flight.total_size && req_end >= req_start {
                        let offset = (req_start - tail_start) as usize;
                        if offset < tail.len() {
                            let end_in_tail = (req_end - tail_start) as usize;
                            let end_clamped = end_in_tail.min(tail.len() - 1);
                            let slice = tail.slice(offset..=end_clamped);
                            let slice_len = slice.len();
                            let end_byte = req_start + slice_len as u64 - 1;

                            METRICS.hits_tail.fetch_add(1, Ordering::Relaxed);
                            METRICS.bytes_served_lan.fetch_add(slice_len as u64, Ordering::Relaxed);

                            info!(
                                %item_id,
                                req_start,
                                end_byte,
                                slice_len,
                                total_size = in_flight.total_size,
                                "[FAST-START] Buffer hit (tail/cues): serving {} bytes locally over LAN",
                                slice_len
                            );

                            let resp = Response::builder()
                                .status(StatusCode::PARTIAL_CONTENT)
                                .header(header::CONTENT_TYPE, &in_flight.content_type)
                                .header(
                                    header::CONTENT_RANGE,
                                    format!("bytes {}-{}/{}", req_start, end_byte, in_flight.total_size),
                                )
                                .header(header::CONTENT_LENGTH, slice_len.to_string())
                                .header(header::ACCEPT_RANGES, "bytes")
                                .header("X-Remux-FastStart", "hit-tail")
                                .body(Body::from(slice))
                                .unwrap();

                            return Some(resp);
                        }
                    }
                }
            }

            METRICS.redirects_issued.fetch_add(1, Ordering::Relaxed);
            info!(
                %item_id,
                range = %rng,
                direct_cdn = %in_flight.direct_cdn_url,
                "[FAST-START] In-flight tail redirect: issuing 307 directly to Debrid CDN"
            );
            let mut redirect = Redirect::temporary(&in_flight.direct_cdn_url).into_response();
            redirect.headers_mut().insert("X-Remux-FastStart", HeaderValue::from_static("redirect-direct-cdn"));
            Some(redirect)
        }

        // Case 4: Any other range (continuous playback past head, seek)
        Some(rng) => {
            METRICS.redirects_issued.fetch_add(1, Ordering::Relaxed);
            info!(
                %item_id,
                range = %rng,
                direct_cdn = %in_flight.direct_cdn_url,
                "[FAST-START] Direct CDN handoff: issuing 307 directly to CDN edge (0 Mbps server upload)"
            );
            let mut redirect = Redirect::temporary(&in_flight.direct_cdn_url).into_response();
            redirect.headers_mut().insert("X-Remux-FastStart", HeaderValue::from_static("redirect-direct-cdn"));
            Some(redirect)
        }
    }
}

/// Returns true when the incoming request originated from outside the local network
/// (e.g. through a Cloudflare Tunnel or public WAN).
pub fn is_remote_client(headers: &HeaderMap) -> bool {
    // 1. Cloudflare Tunnel (cloudflared) injects cf-connecting-ip or cf-ray for external traffic
    if headers.contains_key("cf-connecting-ip") || headers.contains_key("cf-ray") {
        return true;
    }

    // 2. Check X-Forwarded-For or X-Real-IP if present: if it parses to a valid client IP,
    //    decide immediately (so LAN clients behind a local reverse proxy with a domain Host
    //    are NOT falsely classified as remote).
    for hdr in ["x-forwarded-for", "x-real-ip"] {
        if let Some(val) = headers.get(hdr).and_then(|h| h.to_str().ok()) {
            if let Some(first_ip) = val.split(',').next().map(str::trim) {
                if let Ok(ip) = first_ip.parse::<std::net::IpAddr>() {
                    let canon = ip.to_canonical();
                    return !canon.is_loopback() && !is_private_ip(&canon);
                }
            }
        }
    }

    // 3. Check Host header — strip IPv6 brackets `[::1]:3000` properly before IP / domain check
    if let Some(host) = headers.get(header::HOST).and_then(|h| h.to_str().ok()) {
        let host_name = if let Some(rest) = host.strip_prefix('[') {
            rest.split(']').next().unwrap_or(host)
        } else {
            host.split(':').next().unwrap_or(host)
        };
        if let Ok(ip) = host_name.parse::<std::net::IpAddr>() {
            let canon = ip.to_canonical();
            return !canon.is_loopback() && !is_private_ip(&canon);
        }
        let lower = host_name.to_ascii_lowercase();
        // Single-label hostnames (e.g. "localhost", "nas", "remux") and local domains are LAN
        if !lower.contains('.')
            || lower.ends_with(".local")
            || lower.ends_with(".lan")
            || lower.ends_with(".home")
            || lower.ends_with(".home.arpa")
        {
            return false;
        }
        return true;
    }

    false
}

fn is_private_ip(ip: &std::net::IpAddr) -> bool {
    match ip.to_canonical() {
        std::net::IpAddr::V4(v4) => v4.is_private() || v4.is_link_local(),
        std::net::IpAddr::V6(v6) => {
            let seg0 = v6.segments()[0];
            // ULA (fc00::/7) or link-local (fe80::/10)
            (seg0 & 0xfe00) == 0xfc00 || (seg0 & 0xffc0) == 0xfe80
        }
    }
}

/// Returns true when a resolved Debrid CDN URL locks download tokens to the requesting IP
/// (e.g. Real-Debrid / AllDebrid), requiring remote WAN clients to hit the upstream resolver
/// URL so a token is minted for their IP. Token-authenticated CDNs like TorBox (`tb-cdn.earth`)
/// are not IP-locked and can use the cached `direct_cdn_url` directly.
fn is_ip_locked_cdn_url(cdn_url: &str) -> bool {
    let Ok(parsed) = url::Url::parse(cdn_url) else {
        return false;
    };
    let Some(host) = parsed.host_str() else {
        return false;
    };
    let h = host.to_ascii_lowercase();
    h.ends_with(".download.real-debrid.com")
        || h.ends_with(".real-debrid.com")
        || h.ends_with(".rdeb.io")
        || h.ends_with(".debrid.it")
        || h.ends_with(".alldebrid.com")
}

fn build_remote_direct_handoff(
    item_id: Uuid,
    original_url: &str,
    direct_cdn_url: &str,
    token_expired: bool,
    range_header: Option<&str>,
) -> Response {
    METRICS.redirects_issued.fetch_add(1, Ordering::Relaxed);
    let must_use_resolver = token_expired || is_ip_locked_cdn_url(direct_cdn_url);
    let (target_url, tag) = if must_use_resolver {
        (original_url, "remote-direct-resolver")
    } else {
        (direct_cdn_url, "redirect-direct-cdn")
    };
    info!(
        %item_id,
        range = range_header.unwrap_or("none"),
        target = %target_url,
        token_expired,
        "[FAST-START] Remote client fast handoff: bypassing local RAM upload and issuing 307 to CDN"
    );
    let mut redirect = Redirect::temporary(target_url).into_response();
    redirect
        .headers_mut()
        .insert("X-Remux-FastStart", HeaderValue::from_static(tag));
    redirect
}

/// Evaluates an incoming stream playback request from a media player (e.g. Infuse)
/// and handles it via local fast-start cache or direct CDN handoff.
///
/// Returns `Some(Response)` if handled, or `None` if the stream is not cached.
pub async fn handle_playback_request(
    item_id: Uuid,
    url: &str,
    headers: &HeaderMap,
    upstream_headers: Option<&HashMap<String, String>>,
) -> Option<Response> {
    let is_remote = is_remote_client(headers);
    let range_header = headers.get(header::RANGE).and_then(|v| v.to_str().ok());

    if is_force_direct_307() {
        if let Some(entry) = FAST_START_CACHE.get(url) {
            let token_expired = entry.created_at.elapsed() >= Duration::from_secs(3600);
            let must_use_resolver =
                token_expired || (is_remote && is_ip_locked_cdn_url(&entry.direct_cdn_url));
            if !must_use_resolver {
                METRICS.redirects_issued.fetch_add(1, Ordering::Relaxed);
                info!(
                    %item_id,
                    direct_cdn = %entry.direct_cdn_url,
                    is_remote,
                    "[FAST-START] FORCE_DIRECT_307=true: immediately issuing 307 directly to Debrid CDN from byte 0"
                );
                let mut redirect = Redirect::temporary(&entry.direct_cdn_url).into_response();
                redirect.headers_mut().insert(
                    "X-Remux-FastStart",
                    HeaderValue::from_static("force-direct-307"),
                );
                return Some(redirect);
            }
        }
        if is_remote {
            METRICS.redirects_issued.fetch_add(1, Ordering::Relaxed);
            info!(
                %item_id,
                %url,
                "[FAST-START] FORCE_DIRECT_307=true (remote): redirecting directly to upstream resolver (307)"
            );
            let mut redirect = Redirect::temporary(url).into_response();
            redirect.headers_mut().insert(
                "X-Remux-FastStart",
                HeaderValue::from_static("remote-direct-resolver"),
            );
            return Some(redirect);
        }
        let url_str = url.to_string();
        let headers_clone = upstream_headers.cloned().unwrap_or_default();
        tokio::spawn(async move {
            prewarm_fast_start(Some(item_id), &url_str, &headers_clone, PrewarmPriority::High, None, None).await;
        });
        METRICS.redirects_issued.fetch_add(1, Ordering::Relaxed);
        info!(
            %item_id,
            %url,
            "[FAST-START] FORCE_DIRECT_307=true (uncached): redirecting immediately to upstream stream (307)"
        );
        let mut redirect = Redirect::temporary(url).into_response();
        redirect.headers_mut().insert(
            "X-Remux-FastStart",
            HeaderValue::from_static("force-direct-307-uncached"),
        );
        return Some(redirect);
    }

    if is_browser_video_request(headers) {
        if let Some(entry) = FAST_START_CACHE.get(url) {
            let token_expired = entry.created_at.elapsed() >= Duration::from_secs(3600);
            let must_use_resolver =
                token_expired || (is_remote && is_ip_locked_cdn_url(&entry.direct_cdn_url));
            if !must_use_resolver {
                info!(
                    %item_id,
                    direct_cdn = %entry.direct_cdn_url,
                    is_remote,
                    "[FAST-START] Browser <video> request detected; issuing consistent 307 to direct CDN from byte 0"
                );
                let mut redirect = Redirect::temporary(&entry.direct_cdn_url).into_response();
                redirect.headers_mut().insert(
                    "X-Remux-FastStart",
                    HeaderValue::from_static("browser-direct-cdn"),
                );
                return Some(redirect);
            }
        }
        if is_remote {
            let mut redirect = Redirect::temporary(url).into_response();
            redirect.headers_mut().insert(
                "X-Remux-FastStart",
                HeaderValue::from_static("remote-direct-resolver-browser"),
            );
            return Some(redirect);
        }
        info!(
            %item_id,
            "[FAST-START] Browser <video> request detected (uncached); falling back to standard 307 redirect"
        );
        return None;
    }

    // 1. Direct lookup by exact URL to avoid stream hijacking if the user selected a different version/source
    let mut entry = FAST_START_CACHE.get(url);

    // 2. If not cached, check if stream is currently in-flight to stream chunks in real-time (LAN)
    //    or hand off immediately to the resolved CDN edge (Remote).
    if entry.is_none() {
        if let Some(in_flight) = IN_FLIGHT_PREWARMS.get(url) {
            if is_remote {
                return Some(build_remote_direct_handoff(
                    item_id,
                    url,
                    &in_flight.direct_cdn_url,
                    false,
                    range_header,
                ));
            }
            if let Some(resp) = serve_from_in_flight(item_id, in_flight.value(), headers).await {
                return Some(resp);
            }
        }

        // If not yet in IN_FLIGHT_PREWARMS but PREWARM_LOCKS is actively held (e.g. awaiting upstream
        // resolver redirect headers), wait via ACTIVE_PREWARM_NOTIFY up to 2.5s so cold starts attach
        // cleanly to the resolved CDN / in-flight RAM stream instead of bailing out at 500ms.
        let url_key = url.to_string();
        if PREWARM_LOCKS.contains_key(&url_key) {
            let deadline = Instant::now() + Duration::from_millis(2500);
            loop {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    break;
                }
                if let Some(notify) = get_active_prewarm_notify(url) {
                    let notified = notify.notified();
                    tokio::pin!(notified);
                    // Re-check caches AFTER pinning `notified` to prevent lost wakeups
                    if let Some(in_flight) = IN_FLIGHT_PREWARMS.get(url) {
                        if is_remote {
                            return Some(build_remote_direct_handoff(
                                item_id,
                                url,
                                &in_flight.direct_cdn_url,
                                false,
                                range_header,
                            ));
                        }
                        if let Some(resp) = serve_from_in_flight(item_id, in_flight.value(), headers).await {
                            return Some(resp);
                        }
                        break;
                    }
                    if let Some(e) = FAST_START_CACHE.get(url) {
                        entry = Some(e);
                        break;
                    }
                    if !PREWARM_LOCKS.contains_key(&url_key) {
                        break;
                    }
                    let _ = tokio::time::timeout(remaining, &mut notified).await;
                } else {
                    if let Some(in_flight) = IN_FLIGHT_PREWARMS.get(url) {
                        if is_remote {
                            return Some(build_remote_direct_handoff(
                                item_id,
                                url,
                                &in_flight.direct_cdn_url,
                                false,
                                range_header,
                            ));
                        }
                        if let Some(resp) = serve_from_in_flight(item_id, in_flight.value(), headers).await {
                            return Some(resp);
                        }
                        break;
                    }
                    if let Some(e) = FAST_START_CACHE.get(url) {
                        entry = Some(e);
                        break;
                    }
                    if !PREWARM_LOCKS.contains_key(&url_key) {
                        break;
                    }
                    tokio::time::sleep(remaining.min(Duration::from_millis(25))).await;
                }
            }
            // Final check in case registration/completion happened right as the wait ended
            if entry.is_none() {
                if let Some(in_flight) = IN_FLIGHT_PREWARMS.get(url) {
                    if is_remote {
                        return Some(build_remote_direct_handoff(
                            item_id,
                            url,
                            &in_flight.direct_cdn_url,
                            false,
                            range_header,
                        ));
                    }
                    if let Some(resp) = serve_from_in_flight(item_id, in_flight.value(), headers).await {
                        return Some(resp);
                    }
                }
                if let Some(e) = FAST_START_CACHE.get(url) {
                    entry = Some(e);
                }
            }
        }

        if entry.is_none() {
            // For remote clients, do not stall on uncached items: immediately 307 to upstream resolver
            if is_remote {
                METRICS.redirects_issued.fetch_add(1, Ordering::Relaxed);
                info!(
                    %item_id,
                    %url,
                    "[FAST-START] Remote client request uncached: redirecting directly to upstream resolver from byte 0 (307)"
                );
                let mut redirect = Redirect::temporary(url).into_response();
                redirect.headers_mut().insert(
                    "X-Remux-FastStart",
                    HeaderValue::from_static("remote-direct-resolver-uncached"),
                );
                return Some(redirect);
            }

            // Uncached and nothing in-flight (LAN): don't block this request.
            // Spawn pre-warm in background for this exact URL so subsequent seeks benefit, but immediately fall through to 307 redirect.
            info!(%item_id, %url, "[FAST-START] Uncached & not in-flight: spawning background pre-warm and immediately falling back to direct redirect");
            let url_str = url.to_string();
            let headers_clone = upstream_headers.cloned().unwrap_or_default();
            tokio::spawn(async move {
                prewarm_fast_start(Some(item_id), &url_str, &headers_clone, PrewarmPriority::High, None, None).await;
            });
        }
    }

    let entry = entry?;

    if entry.total_size < 2 * 1024 * 1024 || crate::stream::is_error_video_url(&entry.direct_cdn_url) {
        warn!(
            %item_id,
            direct_cdn = %entry.direct_cdn_url,
            total_size = entry.total_size,
            "[FAST-START] Refusing to serve cached error placeholder video from RAM; invalidating"
        );
        FAST_START_CACHE.invalidate(url);
        return None;
    }

    // Remote WAN clients (e.g. via Cloudflare Tunnel) bypass home ISP upload & tunnel throttling
    // by jumping directly to the pre-resolved Debrid CDN edge from byte 0 (for head, tail, and resume),
    // while LAN clients continue streaming head/resume buffers at Gigabit LAN speeds from RAM.
    if is_remote {
        let token_expired = entry.created_at.elapsed() >= Duration::from_secs(3600);
        return Some(build_remote_direct_handoff(
            item_id,
            url,
            &entry.direct_cdn_url,
            token_expired,
            range_header,
        ));
    }

    match range_header {
        // Case 1: Initial probe / head sub-range request (any Range starting inside the cached head buffer)
        Some(rng) if is_head_range(rng, entry.head_bytes.len(), entry.total_size) => {
            METRICS.hits_head.fetch_add(1, Ordering::Relaxed);
            let head_len = entry.head_bytes.len();
            let (req_start, req_end) =
                parse_range_bounds(rng, entry.total_size).unwrap_or((0, entry.total_size.saturating_sub(1)));

            let start_idx = req_start as usize;
            let max_chunk_end = (start_idx + FAST_START_HEAD_SERVE_CHUNK_SIZE).saturating_sub(1);
            let end_idx = (req_end as usize)
                .min(max_chunk_end)
                .min(head_len.saturating_sub(1));
            let slice = entry.head_bytes.slice(start_idx..=end_idx);
            let slice_len = slice.len();
            let end_offset = req_start + (slice_len as u64).saturating_sub(1);

            METRICS.bytes_served_lan.fetch_add(slice_len as u64, Ordering::Relaxed);

            info!(
                %item_id,
                req_start,
                end_offset,
                slice_len,
                head_len,
                total_size = entry.total_size,
                "[FAST-START] Buffer hit (head): serving {} bytes locally over LAN (chunked)",
                slice_len
            );

            let resp = Response::builder()
                .status(StatusCode::PARTIAL_CONTENT)
                .header(header::CONTENT_TYPE, &entry.content_type)
                .header(
                    header::CONTENT_RANGE,
                    format!("bytes {}-{}/{}", req_start, end_offset, entry.total_size),
                )
                .header(header::CONTENT_LENGTH, slice_len.to_string())
                .header(header::ACCEPT_RANGES, "bytes")
                .header("X-Remux-FastStart", "hit-head")
                .body(Body::from(slice))
                .unwrap();

            Some(resp)
        }

        // Case 2: No Range header specified at all (treat as head probe)
        None => {
            let head_len = entry.head_bytes.len();
            if head_len == 0 {
                return None;
            }
            METRICS.hits_head.fetch_add(1, Ordering::Relaxed);
            let chunk_len = head_len.min(FAST_START_HEAD_SERVE_CHUNK_SIZE);
            let slice = entry.head_bytes.slice(0..chunk_len);
            let end_offset = (chunk_len as u64).saturating_sub(1);
            METRICS.bytes_served_lan.fetch_add(chunk_len as u64, Ordering::Relaxed);

            info!(
                %item_id,
                chunk_len,
                head_len,
                total_size = entry.total_size,
                "[FAST-START] Buffer hit (no range): serving {} head bytes locally over LAN (chunked)",
                chunk_len
            );

            let resp = Response::builder()
                .status(StatusCode::PARTIAL_CONTENT)
                .header(header::CONTENT_TYPE, &entry.content_type)
                .header(
                    header::CONTENT_RANGE,
                    format!("bytes 0-{}/{}", end_offset, entry.total_size),
                )
                .header(header::CONTENT_LENGTH, chunk_len.to_string())
                .header(header::ACCEPT_RANGES, "bytes")
                .header("X-Remux-FastStart", "hit-head")
                .body(Body::from(slice))
                .unwrap();

            Some(resp)
        }

        // Case 3: Tail / MKV Cues index probe
        Some(rng) if is_tail_range(rng, entry.total_size) => {
            if let Some(ref tail) = entry.tail_bytes {
                let tail_len = tail.len() as u64;
                let tail_start = entry.total_size.saturating_sub(tail_len);
                if let Some((req_start, req_end)) = parse_range_bounds(rng, entry.total_size) {
                    if req_start >= tail_start && req_start < entry.total_size && req_end >= req_start {
                        let offset = (req_start - tail_start) as usize;
                        if offset < tail.len() {
                            let end_in_tail = (req_end - tail_start) as usize;
                            let end_clamped = end_in_tail.min(tail.len() - 1);
                            let slice = tail.slice(offset..=end_clamped);
                            let slice_len = slice.len();
                            let end_byte = req_start + slice_len as u64 - 1;

                            METRICS.hits_tail.fetch_add(1, Ordering::Relaxed);
                            METRICS.bytes_served_lan.fetch_add(slice_len as u64, Ordering::Relaxed);

                            info!(
                                %item_id,
                                req_start,
                                end_byte,
                                slice_len,
                                total_size = entry.total_size,
                                "[FAST-START] Buffer hit (tail/cues): serving {} bytes locally over LAN",
                                slice_len
                            );

                            let resp = Response::builder()
                                .status(StatusCode::PARTIAL_CONTENT)
                                .header(header::CONTENT_TYPE, &entry.content_type)
                                .header(
                                    header::CONTENT_RANGE,
                                    format!("bytes {}-{}/{}", req_start, end_byte, entry.total_size),
                                )
                                .header(header::CONTENT_LENGTH, slice_len.to_string())
                                .header(header::ACCEPT_RANGES, "bytes")
                                .header("X-Remux-FastStart", "hit-tail")
                                .body(Body::from(slice))
                                .unwrap();

                            return Some(resp);
                        }
                    }
                }
            }

            // Fall through to direct redirect if tail is missing or sub-range not covered
            METRICS.redirects_issued.fetch_add(1, Ordering::Relaxed);
            let token_expired = entry.created_at.elapsed() >= Duration::from_secs(3600);
            let must_use_resolver =
                token_expired || (is_remote && is_ip_locked_cdn_url(&entry.direct_cdn_url));
            let redirect_target = if must_use_resolver { url } else { &entry.direct_cdn_url };
            info!(
                %item_id,
                range = %rng,
                target = %redirect_target,
                is_remote,
                token_expired,
                "[FAST-START] Redirecting tail request (307)"
            );
            let mut redirect = Redirect::temporary(redirect_target).into_response();
            let header_tag = if must_use_resolver { "remote-direct-resolver" } else { "redirect-direct-cdn" };
            redirect.headers_mut().insert("X-Remux-FastStart", HeaderValue::from_static(header_tag));
            Some(redirect)
        }

        // Case 4: Resume buffer hit (when resuming playback from middle)
        Some(rng) if is_resume_range(rng, entry.resume_buffer.as_ref(), entry.total_size) => {
            if let Some((resume_start, ref resume_bytes)) = entry.resume_buffer {
                let (req_start, req_end) = parse_range_bounds(rng, entry.total_size)
                    .unwrap_or((resume_start, entry.total_size.saturating_sub(1)));
                let resume_len = resume_bytes.len() as u64;
                let resume_end = resume_start + resume_len;
                if req_start >= resume_start && req_start < resume_end {
                    let offset = (req_start - resume_start) as usize;
                    let max_chunk_end =
                        (offset + FAST_START_HEAD_SERVE_CHUNK_SIZE).saturating_sub(1);
                    let end_in_resume = (req_end - resume_start) as usize;
                    let end_clamped = end_in_resume
                        .min(max_chunk_end)
                        .min(resume_bytes.len() - 1);
                    let slice = resume_bytes.slice(offset..=end_clamped);
                    let slice_len = slice.len();
                    let end_byte = req_start + slice_len as u64 - 1;

                    METRICS.hits_resume.fetch_add(1, Ordering::Relaxed);
                    METRICS.bytes_served_lan.fetch_add(slice_len as u64, Ordering::Relaxed);

                    info!(
                        %item_id,
                        req_start,
                        end_byte,
                        slice_len,
                        total_size = entry.total_size,
                        "[FAST-START] Buffer hit (resume): serving {} bytes locally over LAN (chunked)",
                        slice_len
                    );

                    let resp = Response::builder()
                        .status(StatusCode::PARTIAL_CONTENT)
                        .header(header::CONTENT_TYPE, &entry.content_type)
                        .header(
                            header::CONTENT_RANGE,
                            format!("bytes {}-{}/{}", req_start, end_byte, entry.total_size),
                        )
                        .header(header::CONTENT_LENGTH, slice_len.to_string())
                        .header(header::ACCEPT_RANGES, "bytes")
                        .header("X-Remux-FastStart", "hit-resume")
                        .body(Body::from(slice))
                        .unwrap();

                    return Some(resp);
                }
            }

            // Fall through to redirect if resume chunk doesn't cover
            METRICS.redirects_issued.fetch_add(1, Ordering::Relaxed);
            let token_expired = entry.created_at.elapsed() >= Duration::from_secs(3600);
            let must_use_resolver =
                token_expired || (is_remote && is_ip_locked_cdn_url(&entry.direct_cdn_url));
            let redirect_target = if must_use_resolver { url } else { &entry.direct_cdn_url };
            info!(
                %item_id,
                range = %rng,
                target = %redirect_target,
                is_remote,
                token_expired,
                "[FAST-START] Resume range miss: redirecting (307)"
            );
            let mut redirect = Redirect::temporary(redirect_target).into_response();
            let header_tag = if must_use_resolver { "remote-direct-resolver" } else { "redirect-direct-cdn" };
            redirect.headers_mut().insert("X-Remux-FastStart", HeaderValue::from_static(header_tag));
            Some(redirect)
        }

        // Case 5: Video cluster stream / seek / continuous playback
        // Issue 307 Temporary Redirect:
        // - For IP-locked Debrid CDNs on remote clients (or expired tokens): to upstream resolver URL
        // - Otherwise (LAN or token-auth CDNs like TorBox): DIRECTLY to Debrid CDN edge (skips resolver hop)
        Some(rng) => {
            METRICS.redirects_issued.fetch_add(1, Ordering::Relaxed);
            let token_expired = entry.created_at.elapsed() >= Duration::from_secs(3600);
            let must_use_resolver =
                token_expired || (is_remote && is_ip_locked_cdn_url(&entry.direct_cdn_url));
            let redirect_target = if must_use_resolver {
                url
            } else {
                &entry.direct_cdn_url
            };
            info!(
                %item_id,
                range = %rng,
                target = %redirect_target,
                is_remote,
                token_expired,
                "[FAST-START] Direct handoff outside buffer: issuing 307 redirect"
            );
            let mut redirect = Redirect::temporary(redirect_target).into_response();
            let header_tag = if must_use_resolver {
                "remote-direct-resolver"
            } else {
                "redirect-direct-cdn"
            };
            redirect.headers_mut().insert("X-Remux-FastStart", HeaderValue::from_static(header_tag));
            Some(redirect)
        }
    }
}

/// Helper to parse byte range bounds from a Range header.
/// Supports both explicit ranges ("bytes=12345-67890", "bytes=12345-")
/// and suffix ranges ("bytes=-262144").
fn parse_range_bounds(range_str: &str, total_size: u64) -> Option<(u64, u64)> {
    if !range_str.starts_with("bytes=") {
        return None;
    }
    let after_prefix = &range_str[6..];
    if after_prefix.starts_with('-') {
        let suffix_len = after_prefix[1..].parse::<u64>().ok()?;
        let start = total_size.saturating_sub(suffix_len);
        let end = total_size.saturating_sub(1);
        Some((start, end))
    } else {
        let mut parts = after_prefix.split('-');
        let start = parts.next()?.parse::<u64>().ok()?;
        let end = parts
            .next()
            .and_then(|s| if s.is_empty() { None } else { s.parse::<u64>().ok() })
            .unwrap_or(total_size.saturating_sub(1));
        Some((start, end))
    }
}

/// Determines whether a Range header starts inside the cached head buffer.
fn is_head_range(range_str: &str, head_len: usize, total_size: u64) -> bool {
    if head_len == 0 {
        return false;
    }
    let Some((start, end)) = parse_range_bounds(range_str, total_size) else {
        return false;
    };
    start < (head_len as u64) && end >= start
}

/// Determines whether a Range header is requesting the tail (MKV Cues / seekhead) region.
fn is_tail_range(range_str: &str, total_size: u64) -> bool {
    let Some((start, _)) = parse_range_bounds(range_str, total_size) else {
        return false;
    };
    let tail_start = total_size.saturating_sub(FAST_START_TAIL_SIZE as u64);
    start >= tail_start && start < total_size
}

/// Determines whether a Range header starts inside the cached resume buffer.
fn is_resume_range(range_str: &str, resume_buf: Option<&(u64, Bytes)>, total_size: u64) -> bool {
    let Some((resume_start, resume_bytes)) = resume_buf else {
        return false;
    };
    let Some((start, _)) = parse_range_bounds(range_str, total_size) else {
        return false;
    };
    let resume_end = resume_start + (resume_bytes.len() as u64);
    start >= *resume_start && start < resume_end
}

fn read_ebml_id(slice: &[u8]) -> Option<(u32, usize)> {
    if slice.is_empty() {
        return None;
    }
    let first = slice[0];
    let num_bytes = first.leading_zeros() as usize + 1;
    if num_bytes > 4 || slice.len() < num_bytes {
        return None;
    }
    let mut val = 0u32;
    for &b in &slice[..num_bytes] {
        val = (val << 8) | (b as u32);
    }
    Some((val, num_bytes))
}

fn read_ebml_vint(slice: &[u8]) -> Option<(u64, usize)> {
    if slice.is_empty() {
        return None;
    }
    let first = slice[0];
    let num_bytes = first.leading_zeros() as usize + 1;
    if num_bytes > 8 || slice.len() < num_bytes {
        return None;
    }
    let mask = if num_bytes >= 8 { 0 } else { (0xFFu8) >> num_bytes };
    let mut val = (first & mask) as u64;
    for &b in &slice[1..num_bytes] {
        val = (val << 8) | (b as u64);
    }
    Some((val, num_bytes))
}

fn read_ebml_uint(slice: &[u8]) -> u64 {
    let mut val = 0u64;
    for &b in slice {
        val = (val << 8) | (b as u64);
    }
    val
}

/// Finds the MKV Segment data offset (where clusters begin).
pub fn find_mkv_segment_data_offset(head_bytes: &[u8]) -> u64 {
    let seg_id = [0x18u8, 0x53, 0x80, 0x67];
    if let Some(pos) = head_bytes.windows(4).position(|w| w == seg_id) {
        let cursor = pos + 4;
        if let Some((_, vint_len)) = read_ebml_vint(&head_bytes[cursor..]) {
            return (cursor + vint_len) as u64;
        }
    }
    52 // Standard fallback for Matroska
}

/// Identifies the Video track number (TrackType == 1) from the Tracks header.
/// If omitted or undetectable, defaults to track 1.
pub fn find_mkv_video_track(head_bytes: &[u8]) -> u64 {
    let tracks_id = [0x16u8, 0x54, 0xAE, 0x6B];
    let mut pos = 0;
    while pos < head_bytes.len() {
        let Some(idx) = head_bytes[pos..].windows(4).position(|w| w == tracks_id) else {
            break;
        };
        let actual_pos = pos + idx;
        pos = actual_pos + 4;

        let Some((tlen, vlen)) = read_ebml_vint(&head_bytes[pos..]) else {
            continue;
        };
        let mut cursor = pos + vlen;
        let end = (cursor + tlen as usize).min(head_bytes.len());

        while cursor < end {
            let Some((eid, eid_len)) = read_ebml_id(&head_bytes[cursor..]) else {
                break;
            };
            cursor += eid_len;
            let Some((elen, evlen)) = read_ebml_vint(&head_bytes[cursor..]) else {
                break;
            };
            cursor += evlen;
            let entry_end = (cursor + elen as usize).min(end);

            if eid == 0xAE {
                // TrackEntry
                let mut sc = cursor;
                let mut tnum = None;
                let mut ttype = None;

                while sc < entry_end {
                    let Some((sub_id, se_len)) = read_ebml_id(&head_bytes[sc..]) else {
                        break;
                    };
                    sc += se_len;
                    let Some((slen, svlen)) = read_ebml_vint(&head_bytes[sc..]) else {
                        break;
                    };
                    sc += svlen;
                    let sub_end = (sc + slen as usize).min(entry_end);

                    if sub_id == 0xD7 {
                        // TrackNumber
                        tnum = Some(read_ebml_uint(&head_bytes[sc..sub_end]));
                    } else if sub_id == 0x83 {
                        // TrackType: 1 = video, 2 = audio, 17 = subtitle
                        ttype = Some(read_ebml_uint(&head_bytes[sc..sub_end]));
                    }
                    sc = sub_end;
                }

                if ttype == Some(1) && tnum.is_some() {
                    return tnum.unwrap();
                }
            }

            cursor = entry_end;
        }
    }
    1
}

/// Extracts the TimestampScale (in nanoseconds, default 1,000,000) from the MKV Info header in head_bytes.
pub fn find_mkv_timestamp_scale(head_bytes: &[u8]) -> u64 {
    let info_id = [0x15u8, 0x49, 0xA9, 0x66];
    let seek_id_prefix = [0x53u8, 0xAB, 0x84];
    let mut pos = 0;
    while pos < head_bytes.len() {
        let Some(idx) = head_bytes[pos..].windows(4).position(|w| w == info_id) else {
            break;
        };
        let actual_pos = pos + idx;
        pos = actual_pos + 4;

        // Skip SeekHead SeekID references (`53 AB 84 15 49 A9 66`)
        if actual_pos >= 3 && head_bytes[actual_pos - 3..actual_pos] == seek_id_prefix {
            continue;
        }

        let Some((ilen, vlen)) = read_ebml_vint(&head_bytes[pos..]) else {
            continue;
        };
        let cursor = pos + vlen;
        let end = (cursor + ilen as usize).min(head_bytes.len());

        let scale_id = [0x2Au8, 0xD7, 0xB1];
        if let Some(sidx) = head_bytes[cursor..end].windows(3).position(|w| w == scale_id) {
            let sc_pos = cursor + sidx + 3;
            if let Some((slen, svlen)) = read_ebml_vint(&head_bytes[sc_pos..]) {
                let sc_data = sc_pos + svlen;
                let sc_end = (sc_data + slen as usize).min(end);
                if sc_end > sc_data {
                    let val = read_ebml_uint(&head_bytes[sc_data..sc_end]);
                    if val > 0 {
                        return val;
                    }
                }
            }
        }
    }
    1_000_000
}

fn parse_mkv_cues_in_buf(
    buf: &[u8],
    seg_offset: u64,
    vid_track: u64,
    target_time_ticks: u64,
) -> Option<u64> {
    let cues_id = [0x1Cu8, 0x53, 0xBB, 0x6B];
    let seek_id_prefix = [0x53u8, 0xAB, 0x84];

    let mut search_pos = 0;
    let cues_pos = loop {
        let idx = buf[search_pos..].windows(4).position(|w| w == cues_id)?;
        let actual = search_pos + idx;
        if actual >= 3 && buf[actual - 3..actual] == seek_id_prefix {
            search_pos = actual + 4;
            continue;
        }
        break actual;
    };

    let mut cursor = cues_pos + 4;
    let (cues_len, vint_len) = read_ebml_vint(&buf[cursor..])?;
    cursor += vint_len;
    let cues_end = (cursor + cues_len as usize).min(buf.len());

    let mut best_cue: Option<(u64, u64)> = None; // (time_ticks, cluster_offset)

    // Iterate through CuePoint elements (ID: 0xBB) strictly within the Cues element bounds
    while cursor < cues_end {
        let Some((pt_id, pt_id_len)) = read_ebml_id(&buf[cursor..cues_end]) else {
            break;
        };
        cursor += pt_id_len;
        let Some((point_len, point_vint_len)) = read_ebml_vint(&buf[cursor..cues_end]) else {
            break;
        };
        cursor += point_vint_len;
        let point_end = (cursor + point_len as usize).min(cues_end);

        if pt_id != 0xBB {
            cursor = point_end;
            continue;
        }

        let mut sub_cursor = cursor;
        let mut cue_time_ms: Option<u64> = None;
        let mut matching_cluster_pos: Option<u64> = None;

        while sub_cursor < point_end {
            let Some((elem_id, elem_id_len)) = read_ebml_id(&buf[sub_cursor..point_end]) else {
                break;
            };
            sub_cursor += elem_id_len;
            let Some((elem_len, elem_vint_len)) = read_ebml_vint(&buf[sub_cursor..point_end]) else {
                break;
            };
            sub_cursor += elem_vint_len;
            let elem_end = (sub_cursor + elem_len as usize).min(point_end);

            if elem_id == 0xB3 {
                // CueTime
                cue_time_ms = Some(read_ebml_uint(&buf[sub_cursor..elem_end]));
            } else if elem_id == 0xB7 {
                // CueTrackPositions: use read_ebml_id so 2-byte IDs like CueBlockNumber (0x5378) don't corrupt offsets
                let mut track_cursor = sub_cursor;
                let mut this_track: Option<u64> = None;
                let mut this_pos: Option<u64> = None;
                while track_cursor < elem_end {
                    let Some((tid, tid_len)) = read_ebml_id(&buf[track_cursor..elem_end]) else {
                        break;
                    };
                    track_cursor += tid_len;
                    let Some((tlen, tvint_len)) = read_ebml_vint(&buf[track_cursor..elem_end]) else {
                        break;
                    };
                    track_cursor += tvint_len;
                    let tend = (track_cursor + tlen as usize).min(elem_end);
                    if tid == 0xF7 {
                        this_track = Some(read_ebml_uint(&buf[track_cursor..tend]));
                    } else if tid == 0xF1 {
                        this_pos = Some(read_ebml_uint(&buf[track_cursor..tend]));
                    }
                    track_cursor = tend;
                }
                if this_track.is_none() || this_track == Some(vid_track) {
                    if let Some(pos) = this_pos {
                        matching_cluster_pos = Some(pos);
                    }
                }
            }
            sub_cursor = elem_end;
        }

        if let (Some(t), Some(c)) = (cue_time_ms, matching_cluster_pos) {
            if t <= target_time_ticks {
                match best_cue {
                    Some((best_t, _)) if t > best_t => {
                        best_cue = Some((t, c));
                    }
                    None => {
                        best_cue = Some((t, c));
                    }
                    _ => {}
                }
            } else if best_cue.is_some() {
                break;
            }
        }

        cursor = point_end;
    }

    // CueClusterPosition is relative to Segment data offset.
    best_cue.map(|(_, pos)| pos + seg_offset)
}

/// Extracts the cluster byte offset for a given timestamp (in seconds) from an MKV head and tail index (Cues).
/// Searches tail_bytes first, then falls back to head_bytes for front-indexed MKVs.
pub fn find_mkv_cue_offset(head_bytes: &[u8], tail_bytes: &[u8], target_time_secs: f64) -> Option<u64> {
    if tail_bytes.len() < 16 && head_bytes.len() < 16 {
        return None;
    }
    let seg_offset = find_mkv_segment_data_offset(head_bytes);
    let vid_track = find_mkv_video_track(head_bytes);
    let time_scale = find_mkv_timestamp_scale(head_bytes);
    let target_time_ticks =
        ((target_time_secs.max(0.0) * 1_000_000_000.0) / (time_scale as f64)) as u64;

    if tail_bytes.len() >= 16 {
        if let Some(offset) = parse_mkv_cues_in_buf(tail_bytes, seg_offset, vid_track, target_time_ticks) {
            return Some(offset);
        }
    }
    if head_bytes.len() >= 16 {
        return parse_mkv_cues_in_buf(head_bytes, seg_offset, vid_track, target_time_ticks);
    }
    None
}

/// Looks up any saved playback position (in seconds) for a user and item.
pub async fn lookup_resume_position(
    db: &sqlx::SqlitePool,
    user_id: Option<Uuid>,
    item_id: Option<Uuid>,
) -> Option<f64> {
    let (Some(user_id), Some(item_id)) = (user_id, item_id) else {
        return None;
    };
    sqlx::query_scalar::<_, i64>(
        "SELECT playback_position FROM user_media_state WHERE user_id = ?1 AND media_id = ?2",
    )
    .bind(user_id)
    .bind(item_id)
    .fetch_optional(db)
    .await
    .ok()
    .flatten()
    .filter(|pos| *pos > 0)
    .map(|pos| pos as f64)
}

/// Purges any cached resume buffer from RAM for a specific media item (e.g. when marked played or finished).
/// Head bytes (32MB) and tail cues are retained so subsequent replays or probes remain 0ms.
pub fn purge_resume_buffer_for_item(item_id: Uuid) {
    for (url, entry) in FAST_START_CACHE.iter() {
        if entry.item_id == Some(item_id) && entry.resume_buffer.is_some() {
            let mut updated = entry.clone();
            updated.resume_buffer = None;
            FAST_START_CACHE.insert(url.to_string(), updated);
            tracing::info!(%url, %item_id, "[FAST-START] Purged resume buffer from RAM for completed item");
        }
    }
}

/// Returns whether an item has an active entry in `FAST_START_CACHE` with valid head bytes
/// that was prewarmed/refreshed within `max_age`.
pub fn is_item_cached_and_fresh(item_id: Uuid, max_age: Duration) -> bool {
    for (_, entry) in FAST_START_CACHE.iter() {
        if entry.item_id == Some(item_id) && !entry.head_bytes.is_empty() {
            if entry.created_at.elapsed() < max_age {
                return true;
            }
        }
    }
    false
}

/// Heuristically detects the MIME type from the file extension in a URL.
fn detect_content_type(url: &str) -> &'static str {
    let path = url.split('?').next().unwrap_or(url);
    if path.ends_with(".mp4") || path.ends_with(".m4v") {
        "video/mp4"
    } else if path.ends_with(".webm") {
        "video/webm"
    } else if path.ends_with(".ts") {
        "video/mp2t"
    } else if path.ends_with(".avi") {
        "video/x-msvideo"
    } else {
        "video/x-matroska"
    }
}

/// Returns a JSON representation of all fast-start metrics and cached items.
pub fn stats_json() -> serde_json::Value {
    let mut cached_items = Vec::new();
    let mut total_ram_bytes: u64 = 0;

    for (url, entry) in FAST_START_CACHE.iter() {
        let head_len = entry.head_bytes.len() as u64;
        let tail_len = entry.tail_bytes.as_ref().map(|b| b.len() as u64).unwrap_or(0);
        let resume_len = entry.resume_buffer.as_ref().map(|(_, b)| b.len() as u64).unwrap_or(0);
        let item_ram = head_len + tail_len + resume_len;
        total_ram_bytes += item_ram;

        cached_items.push(json!({
            "item_id": entry.item_id,
            "original_url": entry.original_url,
            "direct_cdn_host": head_resp_host(&entry.direct_cdn_url),
            "total_file_size_bytes": entry.total_size,
            "head_buffer_bytes": head_len,
            "tail_buffer_bytes": tail_len,
            "resume_buffer_bytes": resume_len,
            "ram_usage_mb": (item_ram as f64) / (1024.0 * 1024.0),
            "age_seconds": entry.created_at.elapsed().as_secs(),
        }));
    }

    json!({
        "metrics": {
            "hits_head": METRICS.hits_head.load(Ordering::Relaxed),
            "hits_tail": METRICS.hits_tail.load(Ordering::Relaxed),
            "hits_resume": METRICS.hits_resume.load(Ordering::Relaxed),
            "redirects_issued": METRICS.redirects_issued.load(Ordering::Relaxed),
            "misses": METRICS.misses.load(Ordering::Relaxed),
            "prewarms_started": METRICS.prewarms_started.load(Ordering::Relaxed),
            "prewarms_high_started": METRICS.prewarms_high_started.load(Ordering::Relaxed),
            "prewarms_low_started": METRICS.prewarms_low_started.load(Ordering::Relaxed),
            "prewarms_succeeded": METRICS.prewarms_succeeded.load(Ordering::Relaxed),
            "prewarms_failed": METRICS.prewarms_failed.load(Ordering::Relaxed),
            "bytes_served_lan": METRICS.bytes_served_lan.load(Ordering::Relaxed),
            "bytes_served_lan_mb": (METRICS.bytes_served_lan.load(Ordering::Relaxed) as f64) / (1024.0 * 1024.0),
        },
        "cache": {
            "force_direct_307": is_force_direct_307(),
            "max_items": FAST_START_MAX_ITEMS,
            "cached_items_count": cached_items.len(),
            "in_flight_count": IN_FLIGHT_PREWARMS.len(),
            "total_ram_bytes": total_ram_bytes,
            "total_ram_mb": (total_ram_bytes as f64) / (1024.0 * 1024.0),
            "ttl_seconds": FAST_START_TTL.as_secs(),
            "items": cached_items,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LOCK: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));

    #[test]
    fn tiered_resume_head_and_buffer_sizing() {
        let url_1080p = "http://cdn.example.com/Show.S01E01.1080p.WEB-DL.mkv";
        let url_4k = "http://cdn.example.com/Lioness.S03E06.2160p.ATV.WEB-DL.mkv";

        assert!(!is_4k_stream(url_1080p, None, None));
        assert!(is_4k_stream(url_4k, None, None));

        assert_eq!(
            calculate_fast_start_resume_head_size(url_1080p, None, None),
            24 * 1024 * 1024
        );
        assert_eq!(
            calculate_fast_start_resume_head_size(url_4k, None, None),
            64 * 1024 * 1024
        );
        assert_eq!(
            calculate_fast_start_head_size(url_1080p, None, None),
            32 * 1024 * 1024
        );
        assert_eq!(
            calculate_fast_start_head_size(url_4k, None, None),
            64 * 1024 * 1024
        );
    }

    #[tokio::test]
    async fn head_subrange_requests_served_from_ram() {
        let _lock = TEST_LOCK.lock().await;
        set_force_direct_307(false);
        let test_url = "http://127.0.0.1:9999/shogun-subrange-test.mkv";
        let head_data: Vec<u8> = (0..16384u32).map(|i| (i % 251) as u8).collect();
        FAST_START_CACHE.insert(
            test_url.to_string(),
            FastStartEntry {
                item_id: Some(Uuid::nil()),
                original_url: test_url.to_string(),
                direct_cdn_url: "http://cdn.example.com/shogun.mkv".to_string(),
                head_bytes: Bytes::from(head_data.clone()),
                tail_bytes: None,
                resume_buffer: None,
                total_size: 100_000_000,
                content_type: "video/x-matroska".to_string(),
                created_at: Instant::now(),
            },
        );

        // Request an open-ended range starting at byte 8154 inside the 16 KB head buffer
        let mut headers = HeaderMap::new();
        headers.insert(header::RANGE, HeaderValue::from_static("bytes=8154-"));
        let resp = handle_playback_request(Uuid::nil(), test_url, &headers, None)
            .await
            .expect("should hit fast-start cache");

        assert_eq!(resp.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(
            resp.headers().get("X-Remux-FastStart").unwrap(),
            "hit-head"
        );
        assert_eq!(
            resp.headers().get(header::CONTENT_RANGE).unwrap(),
            "bytes 8154-16383/100000000"
        );
        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&body_bytes[..], &head_data[8154..16384]);
    }

    #[tokio::test]
    async fn head_requests_chunked_to_16mb_over_lan() {
        let _lock = TEST_LOCK.lock().await;
        set_force_direct_307(false);
        let test_url = "http://127.0.0.1:9999/chunked-16mb-test.mkv";
        let head_size = 36 * 1024 * 1024;
        let head_data: Vec<u8> = vec![0x33; head_size];
        FAST_START_CACHE.insert(
            test_url.to_string(),
            FastStartEntry {
                item_id: Some(Uuid::nil()),
                original_url: test_url.to_string(),
                direct_cdn_url: "http://cdn.example.com/test.mkv".to_string(),
                head_bytes: Bytes::from(head_data.clone()),
                tail_bytes: None,
                resume_buffer: None,
                total_size: 5_000_000_000,
                content_type: "video/x-matroska".to_string(),
                created_at: Instant::now(),
            },
        );

        // 1. Client requests open-ended byte 0: should be capped to first 16 MB chunk
        let mut headers1 = HeaderMap::new();
        headers1.insert(header::RANGE, HeaderValue::from_static("bytes=0-"));
        let resp1 = handle_playback_request(Uuid::nil(), test_url, &headers1, None)
            .await
            .expect("should hit head buffer");

        assert_eq!(resp1.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(
            resp1.headers().get(header::CONTENT_RANGE).unwrap(),
            "bytes 0-16777215/5000000000"
        );
        assert_eq!(
            resp1.headers().get(header::CONTENT_LENGTH).unwrap(),
            "16777216"
        );

        // 2. Client requests second chunk (bytes 16777216-): should return next 16 MB chunk
        let mut headers2 = HeaderMap::new();
        headers2.insert(header::RANGE, HeaderValue::from_static("bytes=16777216-"));
        let resp2 = handle_playback_request(Uuid::nil(), test_url, &headers2, None)
            .await
            .expect("should hit head buffer");

        assert_eq!(resp2.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(
            resp2.headers().get(header::CONTENT_RANGE).unwrap(),
            "bytes 16777216-33554431/5000000000"
        );
        assert_eq!(
            resp2.headers().get(header::CONTENT_LENGTH).unwrap(),
            "16777216"
        );

        // 3. Client requests third chunk (bytes 33554432-): should return remaining 4 MB chunk
        let mut headers3 = HeaderMap::new();
        headers3.insert(header::RANGE, HeaderValue::from_static("bytes=33554432-"));
        let resp3 = handle_playback_request(Uuid::nil(), test_url, &headers3, None)
            .await
            .expect("should hit head buffer");

        assert_eq!(resp3.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(
            resp3.headers().get(header::CONTENT_RANGE).unwrap(),
            "bytes 33554432-37748735/5000000000"
        );
        assert_eq!(
            resp3.headers().get(header::CONTENT_LENGTH).unwrap(),
            "4194304"
        );
    }

    #[tokio::test]
    async fn high_priority_prewarm_overtakes_queued_low_priority_on_same_url() {
        let _lock = TEST_LOCK.lock().await;
        set_force_direct_307(false);

        // Exhaust all 2 permits of the background semaphore
        let p1 = FAST_START_PREWARM_SEMAPHORE.acquire().await.unwrap();
        let p2 = FAST_START_PREWARM_SEMAPHORE.acquire().await.unwrap();

        let server = axum::Router::new().route(
            "/video.mkv",
            axum::routing::get(|| async {
                Response::builder()
                    .status(StatusCode::PARTIAL_CONTENT)
                    .header(header::CONTENT_TYPE, "video/x-matroska")
                    .header(header::CONTENT_RANGE, "bytes 0-15/5000000000")
                    .body(Body::from(vec![0x42u8; 16]))
                    .unwrap()
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, server).await.unwrap();
        });

        let test_url = format!("http://{}/video.mkv", addr);
        let url_low = test_url.clone();

        // Spawn a Low-priority pre-warm; it must wait in the semaphore queue WITHOUT holding PREWARM_LOCKS
        let low_handle = tokio::spawn(async move {
            prewarm_fast_start(None, &url_low, &HashMap::new(), PrewarmPriority::Low, None, None).await
        });

        // Give Low task a moment to enter the semaphore queue
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(QUEUED_PREWARMS.contains(&test_url));
        assert!(
            !PREWARM_LOCKS.contains_key(&test_url),
            "Queued Low-priority task must NOT hold PREWARM_LOCKS before acquiring semaphore permit"
        );

        // Now run a High-priority pre-warm for the EXACT SAME URL while the semaphore is still 100% exhausted!
        let high_ok = tokio::time::timeout(
            Duration::from_millis(500),
            prewarm_fast_start(None, &test_url, &HashMap::new(), PrewarmPriority::High, None, None),
        )
        .await
        .expect("High-priority pre-warm should not be blocked by exhausted background semaphore");
        assert!(high_ok);
        assert!(FAST_START_CACHE.get(&test_url).is_some());
        assert_eq!(
            crate::stream::STREAM_LIVENESS_CACHE.get(&test_url).map(|e| e.alive),
            Some(true)
        );

        // Release background permits; Low task wakes up, sees cache populated, and returns true immediately
        drop(p1);
        drop(p2);
        let low_ok = low_handle.await.unwrap();
        assert!(low_ok);
    }

    #[tokio::test]
    async fn force_direct_307_bypasses_ram_buffer() {
        let _lock = TEST_LOCK.lock().await;
        let test_url = "http://127.0.0.1:9999/force-307-test.mkv";
        let cdn_url = "http://cdn.example.com/direct-debrid.mkv";
        FAST_START_CACHE.insert(
            test_url.to_string(),
            FastStartEntry {
                item_id: Some(Uuid::nil()),
                original_url: test_url.to_string(),
                direct_cdn_url: cdn_url.to_string(),
                head_bytes: Bytes::from(vec![0xAA; 1024]),
                tail_bytes: None,
                resume_buffer: None,
                total_size: 1_000_000,
                content_type: "video/x-matroska".to_string(),
                created_at: Instant::now(),
            },
        );

        // When FORCE_DIRECT_307 is enabled, even a request for byte 0 gets a 307 redirect
        set_force_direct_307(true);
        let mut headers = HeaderMap::new();
        headers.insert(header::RANGE, HeaderValue::from_static("bytes=0-"));
        let resp = handle_playback_request(Uuid::nil(), test_url, &headers, None)
            .await
            .expect("should handle request");

        assert_eq!(resp.status(), StatusCode::TEMPORARY_REDIRECT);
        assert_eq!(
            resp.headers().get("X-Remux-FastStart").unwrap(),
            "force-direct-307"
        );
        assert_eq!(
            resp.headers().get(header::LOCATION).unwrap(),
            cdn_url
        );

        // Reset to false
        set_force_direct_307(false);
    }

    #[tokio::test]
    async fn resume_buffer_hit_serves_local_lan_and_direct_307_fallback() {
        let _lock = TEST_LOCK.lock().await;
        set_force_direct_307(false);
        let test_url = "http://127.0.0.1:9999/resume-test.mkv";
        let cdn_url = "http://cdn.example.com/resume-direct.mkv";
        let resume_start: u64 = 5_000_000;
        let resume_data: Vec<u8> = (0..16384u32).map(|i| (i % 251) as u8).collect();

        FAST_START_CACHE.insert(
            test_url.to_string(),
            FastStartEntry {
                item_id: Some(Uuid::nil()),
                original_url: test_url.to_string(),
                direct_cdn_url: cdn_url.to_string(),
                head_bytes: Bytes::from(vec![0x11; 1024]),
                tail_bytes: None,
                resume_buffer: Some((resume_start, Bytes::from(resume_data.clone()))),
                total_size: 10_000_000,
                content_type: "video/x-matroska".to_string(),
                created_at: Instant::now(),
            },
        );

        // 1. Client seeks to byte 5,000,100 (inside the 16KB resume buffer starting at 5,000,000)
        let mut headers = HeaderMap::new();
        headers.insert(header::RANGE, HeaderValue::from_static("bytes=5000100-"));
        let resp = handle_playback_request(Uuid::nil(), test_url, &headers, None)
            .await
            .expect("should hit resume buffer");

        assert_eq!(resp.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(
            resp.headers().get("X-Remux-FastStart").unwrap(),
            "hit-resume"
        );
        assert_eq!(
            resp.headers().get(header::CONTENT_RANGE).unwrap(),
            "bytes 5000100-5016383/10000000"
        );
        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&body_bytes[..], &resume_data[100..]);

        // 2. Client seeks past resume buffer (e.g. byte 5,020,000): should immediately 307 redirect to CDN
        let mut headers_seek = HeaderMap::new();
        headers_seek.insert(header::RANGE, HeaderValue::from_static("bytes=5020000-"));
        let resp_seek = handle_playback_request(Uuid::nil(), test_url, &headers_seek, None)
            .await
            .expect("should redirect to CDN");

        assert_eq!(resp_seek.status(), StatusCode::TEMPORARY_REDIRECT);
        assert_eq!(
            resp_seek.headers().get("X-Remux-FastStart").unwrap(),
            "redirect-direct-cdn"
        );
        assert_eq!(
            resp_seek.headers().get(header::LOCATION).unwrap(),
            cdn_url
        );
    }

    #[tokio::test]
    async fn in_flight_chunk_streaming_serves_live_chunks() {
        let _lock = TEST_LOCK.lock().await;
        set_force_direct_307(false);
        let test_url = "http://127.0.0.1:9999/inflight-live-stream-test.mkv";
        let target_size = 1024 * 1024;
        let total_size = 50 * 1024 * 1024;

        let in_flight = Arc::new(InFlightHead {
            item_id: Some(Uuid::nil()),
            original_url: test_url.to_string(),
            direct_cdn_url: "http://cdn.example.com/inflight.mkv".to_string(),
            total_size,
            target_head_size: target_size,
            content_type: "video/x-matroska".to_string(),
            buffer: RwLock::new(BytesMut::new()),
            tail_bytes: RwLock::new(None),
            notify: tokio::sync::Notify::new(),
            tail_notify: tokio::sync::Notify::new(),
            is_done: AtomicBool::new(false),
            has_error: AtomicBool::new(false),
        });
        IN_FLIGHT_PREWARMS.insert(test_url.to_string(), in_flight.clone());

        // Client requests open-ended byte 0
        let mut headers = HeaderMap::new();
        headers.insert(header::RANGE, HeaderValue::from_static("bytes=0-"));
        let resp = handle_playback_request(Uuid::nil(), test_url, &headers, None)
            .await
            .expect("should hit in-flight streaming");

        assert_eq!(resp.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(
            resp.headers().get("X-Remux-FastStart").unwrap(),
            "hit-head-inflight"
        );
        assert_eq!(
            resp.headers().get(header::CONTENT_RANGE).unwrap().to_str().unwrap(),
            format!("bytes 0-{}/{}", target_size - 1, total_size)
        );

        // Feed chunks into in_flight buffer asynchronously
        let in_flight_producer = in_flight.clone();
        tokio::spawn(async move {
            for i in 0..4 {
                tokio::time::sleep(Duration::from_millis(10)).await;
                {
                    let mut b = in_flight_producer.buffer.write().unwrap();
                    b.extend_from_slice(&vec![i as u8; 256 * 1024]);
                }
                in_flight_producer.notify.notify_waiters();
            }
            in_flight_producer.is_done.store(true, Ordering::Relaxed);
            in_flight_producer.notify.notify_waiters();
        });

        // Drain the streamed body
        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body_bytes.len(), target_size);
        assert_eq!(&body_bytes[0..4], &[0, 0, 0, 0]);
        assert_eq!(&body_bytes[256 * 1024..256 * 1024 + 4], &[1, 1, 1, 1]);

        IN_FLIGHT_PREWARMS.remove(test_url);
    }

    #[tokio::test]
    async fn in_flight_sub_range_does_not_overflow_target_head_size() {
        let _lock = TEST_LOCK.lock().await;
        set_force_direct_307(false);
        let test_url = "http://127.0.0.1:9999/in-flight-overflow-test.mkv";
        let target_size = 32 * 1024 * 1024;
        let total_size = 5_000_000_000u64;

        let in_flight = Arc::new(InFlightHead {
            item_id: Some(Uuid::nil()),
            original_url: test_url.to_string(),
            direct_cdn_url: "http://cdn.example.com/inflight.mkv".to_string(),
            total_size,
            target_head_size: target_size,
            content_type: "video/x-matroska".to_string(),
            buffer: RwLock::new(BytesMut::new()),
            tail_bytes: RwLock::new(None),
            notify: tokio::sync::Notify::new(),
            tail_notify: tokio::sync::Notify::new(),
            is_done: AtomicBool::new(false),
            has_error: AtomicBool::new(false),
        });
        IN_FLIGHT_PREWARMS.insert(test_url.to_string(), in_flight.clone());

        // Client requests a range starting at 28 MB (inside 32 MB target_head_size)
        // With 16 MB chunking, 28 MB + 16 MB = 44 MB, which would overshoot 32 MB!
        // It must be clamped to byte 32 MB - 1 = 33554431.
        let mut headers = HeaderMap::new();
        headers.insert(header::RANGE, HeaderValue::from_static("bytes=29360128-")); // 28 MB
        let resp = handle_playback_request(Uuid::nil(), test_url, &headers, None)
            .await
            .expect("should hit in-flight streaming");

        assert_eq!(resp.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(
            resp.headers().get(header::CONTENT_RANGE).unwrap().to_str().unwrap(),
            "bytes 29360128-33554431/5000000000"
        );
        assert_eq!(
            resp.headers().get(header::CONTENT_LENGTH).unwrap().to_str().unwrap(),
            "4194304" // exactly 4 MB remaining, not 16 MB!
        );

        IN_FLIGHT_PREWARMS.remove(test_url);
    }

    #[tokio::test]
    async fn active_prewarm_notify_guard_cleans_up_on_early_return() {
        let _lock = TEST_LOCK.lock().await;
        let test_url = "http://127.0.0.1:9999/early-return-notify-test.mkv";
        FAST_START_CACHE.invalidate(test_url);

        {
            let _guard = register_active_prewarm_notify(test_url);
            assert!(
                get_active_prewarm_notify(test_url).is_some(),
                "notify should be registered while guard is alive"
            );
            // Simulate nested registration inside prewarm_fast_start
            {
                let _inner_guard = register_active_prewarm_notify(test_url);
                assert!(get_active_prewarm_notify(test_url).is_some());
            }
            // Outer guard still alive
            assert!(get_active_prewarm_notify(test_url).is_some());
        }
        // Both guards dropped — entry must be removed from ACTIVE_PREWARM_NOTIFY
        assert!(
            get_active_prewarm_notify(test_url).is_none(),
            "notify entry must be removed after all guards drop"
        );
    }

    #[tokio::test]
    async fn liveness_refresh_updates_torbox_cdn_url_in_place_and_evicts_on_dead_link() {
        let _lock = TEST_LOCK.lock().await;
        set_force_direct_307(false);

        let should_fail = Arc::new(AtomicBool::new(false));
        let should_fail_clone = should_fail.clone();

        let server = axum::Router::new()
            .route(
                "/strem/video.mkv",
                axum::routing::get(move |req: axum::extract::Request| {
                    let fail = should_fail_clone.load(Ordering::SeqCst);
                    let host = req
                        .headers()
                        .get(header::HOST)
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("127.0.0.1")
                        .to_string();
                    async move {
                        if fail {
                            Response::builder()
                                .status(StatusCode::NOT_FOUND)
                                .body(Body::empty())
                                .unwrap()
                        } else {
                            let new_cdn = format!("http://{}/dld/{}", host, Uuid::new_v4());
                            Redirect::temporary(&new_cdn).into_response()
                        }
                    }
                }),
            )
            .route(
                "/dld/{id}",
                axum::routing::get(|| async {
                    Response::builder()
                        .status(StatusCode::PARTIAL_CONTENT)
                        .header(header::CONTENT_TYPE, "video/x-matroska")
                        .header(header::CONTENT_RANGE, "bytes 0-7/5000000000")
                        .body(Body::from(vec![0x1A, 0x45, 0xDF, 0xA3, 0x01, 0x02, 0x03, 0x04]))
                        .unwrap()
                }),
            );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, server).await.unwrap();
        });

        let test_url = format!("http://{}/strem/video.mkv", addr);
        let initial_cdn = format!("http://{}/dld/old-ticket-uuid", addr);
        FAST_START_CACHE.insert(
            test_url.clone(),
            FastStartEntry {
                item_id: Some(Uuid::nil()),
                original_url: test_url.clone(),
                direct_cdn_url: initial_cdn.clone(),
                total_size: 5_000_000_000,
                head_bytes: Bytes::from(vec![0x42u8; 4096]),
                tail_bytes: Some(Bytes::from(vec![0x99u8; 512])),
                resume_buffer: Some((100_000_000, Bytes::from(vec![0x77u8; 2048]))),
                content_type: "video/x-matroska".to_string(),
                created_at: Instant::now() - Duration::from_secs(7200),
            },
        );

        let descriptor = crate::stream::StreamDescriptor::Http {
            url: test_url.clone(),
            request_headers: HashMap::new(),
            response_headers: HashMap::new(),
        };

        // 1. 8-byte liveness check succeeds with a brand-new /dld/<uuid> path:
        // RAM buffers must remain intact while direct_cdn_url and created_at are refreshed!
        assert!(descriptor.refresh_liveness().await);
        let refreshed = FAST_START_CACHE
            .get(&test_url)
            .expect("RAM buffer must not be evicted when TorBox mints a new /dld/<uuid> ticket");
        assert_ne!(refreshed.direct_cdn_url, initial_cdn);
        assert!(refreshed.direct_cdn_url.contains("/dld/"));
        assert_eq!(refreshed.head_bytes.len(), 4096);
        assert!(refreshed.resume_buffer.is_some());
        assert!(refreshed.created_at.elapsed() < Duration::from_secs(5));

        // 2. Simulate liveness TTL expiring (>1h later) and upstream link dying (404):
        // 8-byte liveness check fails and MUST evict the dead RAM buffer!
        crate::stream::STREAM_LIVENESS_CACHE.invalidate(&test_url);
        should_fail.store(true, Ordering::SeqCst);
        assert!(!descriptor.refresh_liveness().await);
        assert!(
            FAST_START_CACHE.get(&test_url).is_none(),
            "dead upstream link must evict stale RAM buffer"
        );
    }

    #[tokio::test]
    async fn resume_buffer_is_chunked_to_16mb_and_remote_torbox_uses_direct_cdn() {
        let _lock = TEST_LOCK.lock().await;
        set_force_direct_307(false);

        let resolver_url = "https://meteor.example.com/play/stream.mkv";
        let torbox_cdn = "https://nexus.erth.tb-cdn.earth/dld/12345678-aaaa-bbbb-cccc-1234567890ab?token=abc";
        let resume_start: u64 = 1_000_000_000;
        let resume_size = 36 * 1024 * 1024; // 36 MB -> 16 MB + 16 MB + 4 MB
        FAST_START_CACHE.insert(
            resolver_url.to_string(),
            FastStartEntry {
                item_id: Some(Uuid::nil()),
                original_url: resolver_url.to_string(),
                direct_cdn_url: torbox_cdn.to_string(),
                head_bytes: Bytes::from(vec![0x11; 4096]),
                tail_bytes: None,
                resume_buffer: Some((resume_start, Bytes::from(vec![0x55; resume_size]))),
                total_size: 8_000_000_000,
                content_type: "video/x-matroska".to_string(),
                created_at: Instant::now(),
            },
        );

        // 1. LAN client open-ended resume request must be chunked to 16 MB (not all 36 MB!)
        let mut lan_headers = HeaderMap::new();
        lan_headers.insert(header::RANGE, HeaderValue::from_static("bytes=1000000000-"));
        let resp1 = handle_playback_request(Uuid::nil(), resolver_url, &lan_headers, None)
            .await
            .expect("LAN client should hit resume buffer");
        assert_eq!(resp1.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(
            resp1.headers().get(header::CONTENT_LENGTH).unwrap(),
            "16777216"
        );
        assert_eq!(
            resp1.headers().get(header::CONTENT_RANGE).unwrap(),
            "bytes 1000000000-1016777215/8000000000"
        );

        // 2. Remote client (Cloudflare Tunnel) bypasses RAM upload from byte 0 AND on resume,
        //    redirecting 307 straight to pre-resolved tb-cdn.earth!
        let mut remote_head_headers = HeaderMap::new();
        remote_head_headers.insert("cf-connecting-ip", HeaderValue::from_static("97.126.3.115"));
        remote_head_headers.insert(header::RANGE, HeaderValue::from_static("bytes=0-"));
        let resp_remote_head =
            handle_playback_request(Uuid::nil(), resolver_url, &remote_head_headers, None)
                .await
                .expect("remote head should 307 straight to TorBox CDN");
        assert_eq!(resp_remote_head.status(), StatusCode::TEMPORARY_REDIRECT);
        assert_eq!(
            resp_remote_head.headers().get(header::LOCATION).unwrap(),
            torbox_cdn
        );
        assert_eq!(
            resp_remote_head.headers().get("X-Remux-FastStart").unwrap(),
            "redirect-direct-cdn"
        );

        let mut handoff_headers = HeaderMap::new();
        handoff_headers.insert("cf-connecting-ip", HeaderValue::from_static("97.126.3.115"));
        handoff_headers.insert(header::RANGE, HeaderValue::from_static("bytes=1000000000-"));
        let resp_tb = handle_playback_request(Uuid::nil(), resolver_url, &handoff_headers, None)
            .await
            .expect("remote resume should redirect directly to TorBox CDN");
        assert_eq!(resp_tb.status(), StatusCode::TEMPORARY_REDIRECT);
        assert_eq!(
            resp_tb.headers().get(header::LOCATION).unwrap(),
            torbox_cdn
        );
        assert_eq!(
            resp_tb.headers().get("X-Remux-FastStart").unwrap(),
            "redirect-direct-cdn"
        );

        // 3. IP-locked Real-Debrid CDN on remote client must still fall back to resolver URL
        let rd_resolver = "https://meteor.example.com/play/rd-stream.mkv";
        FAST_START_CACHE.insert(
            rd_resolver.to_string(),
            FastStartEntry {
                item_id: Some(Uuid::nil()),
                original_url: rd_resolver.to_string(),
                direct_cdn_url: "https://23.download.real-debrid.com/d/ABCDEF/movie.mkv".to_string(),
                head_bytes: Bytes::from(vec![0x11; 4096]),
                tail_bytes: None,
                resume_buffer: None,
                total_size: 8_000_000_000,
                content_type: "video/x-matroska".to_string(),
                created_at: Instant::now(),
            },
        );
        let resp_rd = handle_playback_request(Uuid::nil(), rd_resolver, &handoff_headers, None)
            .await
            .expect("should redirect RD remote client to resolver");
        assert_eq!(resp_rd.status(), StatusCode::TEMPORARY_REDIRECT);
        assert_eq!(
            resp_rd.headers().get(header::LOCATION).unwrap(),
            rd_resolver
        );
        assert_eq!(
            resp_rd.headers().get("X-Remux-FastStart").unwrap(),
            "remote-direct-resolver"
        );
    }

    #[tokio::test]
    async fn normalizes_item_id_for_freshness_and_purge_and_waits_for_slow_prewarm_headers() {
        let _lock = TEST_LOCK.lock().await;
        set_force_direct_307(false);

        // 1. Entry initially tagged with a child stream UUID gets normalized to episode UUID
        let stream_uuid = Uuid::new_v4();
        let episode_uuid = Uuid::new_v4();
        let cached_url = format!("https://meteor.example.com/play/{}.mkv", episode_uuid);
        FAST_START_CACHE.insert(
            cached_url.clone(),
            FastStartEntry {
                item_id: Some(stream_uuid),
                original_url: cached_url.clone(),
                direct_cdn_url: "https://nexus.erth.tb-cdn.earth/dld/test-ticket".to_string(),
                head_bytes: Bytes::from(vec![0x11; 4096]),
                tail_bytes: None,
                resume_buffer: Some((1_000_000, Bytes::from(vec![0x22; 4096]))),
                total_size: 4_000_000_000,
                content_type: "video/x-matroska".to_string(),
                created_at: Instant::now(),
            },
        );
        assert!(!is_item_cached_and_fresh(episode_uuid, Duration::from_secs(1800)));

        // Re-invoking prewarm with the episode UUID normalizes item_id in-place
        assert!(
            prewarm_fast_start(
                Some(episode_uuid),
                &cached_url,
                &HashMap::new(),
                PrewarmPriority::Low,
                None,
                None,
            )
            .await
        );
        assert!(is_item_cached_and_fresh(episode_uuid, Duration::from_secs(1800)));
        purge_resume_buffer_for_item(episode_uuid);
        assert!(
            FAST_START_CACHE.get(&cached_url).unwrap().resume_buffer.is_none(),
            "resume buffer must be purged using the episode UUID"
        );
        FAST_START_CACHE.invalidate(&cached_url);

        // 2. Slow upstream resolver (>500ms before headers) must wake concurrent remote playback via ACTIVE_PREWARM_NOTIFY
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let total_size: usize = 3 * 1024 * 1024;
        let payload = Bytes::from(vec![0xAB; total_size]);
        let payload_clone = payload.clone();
        let app = axum::Router::new().route(
            "/slow-resolve.mkv",
            axum::routing::get(move |headers: HeaderMap| {
                let data = payload_clone.clone();
                async move {
                    let range = headers
                        .get(header::RANGE)
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("bytes=0-1048575");
                    if range.starts_with("bytes=0-") {
                        // Simulate slow StremThru redirect resolution (750ms > old 500ms poll timeout)
                        tokio::time::sleep(Duration::from_millis(750)).await;
                    }
                    let (start, end) = parse_range_bounds(range, data.len() as u64)
                        .unwrap_or((0, (data.len() - 1) as u64));
                    let slice = data.slice(start as usize..=end as usize);
                    Response::builder()
                        .status(StatusCode::PARTIAL_CONTENT)
                        .header(header::CONTENT_TYPE, "video/x-matroska")
                        .header(
                            header::CONTENT_RANGE,
                            format!("bytes {}-{}/{}", start, end, data.len()),
                        )
                        .header(header::CONTENT_LENGTH, slice.len().to_string())
                        .body(Body::from(slice))
                        .unwrap()
                }
            }),
        );
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let slow_url = format!("http://{}/slow-resolve.mkv", addr);
        let slow_item_id = Uuid::new_v4();
        let prewarm_url = slow_url.clone();
        let prewarm_task = tokio::spawn(async move {
            prewarm_fast_start(
                Some(slow_item_id),
                &prewarm_url,
                &HashMap::new(),
                PrewarmPriority::Low,
                Some(2 * 1024 * 1024),
                None,
            )
            .await
        });

        // Wait briefly so prewarm_fast_start has acquired PREWARM_LOCKS and is awaiting upstream headers
        tokio::time::sleep(Duration::from_millis(80)).await;

        let mut lan_headers = HeaderMap::new();
        lan_headers.insert(header::RANGE, HeaderValue::from_static("bytes=0-65535"));

        let mut remote_headers = HeaderMap::new();
        remote_headers.insert("cf-connecting-ip", HeaderValue::from_static("97.126.3.115"));
        remote_headers.insert(header::RANGE, HeaderValue::from_static("bytes=0-65535"));

        let (lan_resp, remote_resp) = tokio::join!(
            handle_playback_request(slow_item_id, &slow_url, &lan_headers, None),
            handle_playback_request(slow_item_id, &slow_url, &remote_headers, None),
        );

        let lan_resp = lan_resp.expect("LAN should attach to in-flight prewarm once headers arrive");
        assert_eq!(lan_resp.status(), StatusCode::PARTIAL_CONTENT);
        let fs_hdr = lan_resp
            .headers()
            .get("X-Remux-FastStart")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            fs_hdr == "hit-head-inflight" || fs_hdr == "hit-head",
            "expected LAN RAM hit instead of premature 307, got {fs_hdr}"
        );

        let remote_resp = remote_resp.expect("Remote should wait for active prewarm headers then 307 straight to resolved CDN");
        assert_eq!(remote_resp.status(), StatusCode::TEMPORARY_REDIRECT);
        assert_eq!(
            remote_resp.headers().get("X-Remux-FastStart").unwrap(),
            "redirect-direct-cdn"
        );

        assert!(prewarm_task.await.unwrap());
        FAST_START_CACHE.invalidate(&slow_url);
        server.abort();
    }
}
