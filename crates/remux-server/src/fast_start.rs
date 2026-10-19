use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Redirect, Response};
use bytes::Bytes;
use serde_json::json;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Default head buffer size in bytes for standard 1080p / smaller streams: 16 MB.
pub const FAST_START_DEFAULT_HEAD_SIZE: usize = 16 * 1024 * 1024;

/// 4K / UHD / large stream head buffer size in bytes: 32 MB.
pub const FAST_START_4K_HEAD_SIZE: usize = 32 * 1024 * 1024;

/// Backward-compatible alias for default head size.
pub const FAST_START_HEAD_SIZE: usize = FAST_START_DEFAULT_HEAD_SIZE;

/// Tail buffer size for MKV Cues / seekhead index: 2 MB.
pub const FAST_START_TAIL_SIZE: usize = 2 * 1024 * 1024;

/// Default resume buffer size for standard 1080p / smaller streams: 16 MB.
pub const FAST_START_DEFAULT_RESUME_SIZE: usize = 16 * 1024 * 1024;

/// 4K / UHD / large stream resume buffer size: 32 MB.
pub const FAST_START_4K_RESUME_SIZE: usize = 32 * 1024 * 1024;

/// Determines the ideal Fast-Start head buffer size based on URL clues, file size, or 4K flags.
pub fn calculate_fast_start_head_size(
    url: &str,
    size_bytes: Option<u64>,
    is_4k: Option<bool>,
) -> usize {
    if is_4k == Some(true) {
        return FAST_START_4K_HEAD_SIZE;
    }
    if let Some(bytes) = size_bytes {
        if bytes > 8 * 1024 * 1024 * 1024 {
            return FAST_START_4K_HEAD_SIZE;
        }
    }
    let lower = url.to_ascii_lowercase();
    if lower.contains("2160p")
        || lower.contains("4k")
        || lower.contains("uhd")
        || lower.contains("remux")
    {
        FAST_START_4K_HEAD_SIZE
    } else {
        FAST_START_DEFAULT_HEAD_SIZE
    }
}

/// Maximum number of items held in memory simultaneously.
pub const FAST_START_MAX_ITEMS: usize = 50;

/// Cache TTL matching Remux's LIVENESS_CACHE_TTL (1 hour).
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

/// Fast-start buffer cache, keyed by upstream stream URL.
pub(crate) static FAST_START_CACHE: LazyLock<moka::sync::Cache<String, FastStartEntry>> =
    LazyLock::new(|| {
        moka::sync::Cache::builder()
            .max_capacity(FAST_START_MAX_ITEMS as u64)
            .time_to_live(FAST_START_TTL)
            .build()
    });

/// Tracks URLs currently queued or running in the low-priority background lane to prevent duplicate spawns.
static QUEUED_PREWARMS: LazyLock<dashmap::DashSet<String>> =
    LazyLock::new(dashmap::DashSet::new);

/// Keyed lock held ONLY while actively downloading a URL (after any background semaphore permit is acquired).
static PREWARM_LOCKS: LazyLock<crate::keyed_lock::KeyedLock<String>> =
    LazyLock::new(crate::keyed_lock::KeyedLock::new);

/// Global semaphore limiting concurrent low-priority background pre-warms to 2 simultaneous chunk downloads,
/// leaving bandwidth headroom for high-priority interactive playback requests.
pub static FAST_START_PREWARM_SEMAPHORE: LazyLock<tokio::sync::Semaphore> =
    LazyLock::new(|| tokio::sync::Semaphore::new(2));

/// Dedicated reqwest client for pre-warming (follows HTTP 302/307 redirects to CDN).
static PREWARM_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .expect("failed to build fast-start prewarm client")
});

/// Pre-warms the head buffer (16/32 MB), tail index (2 MB), and resume chunk (16/32 MB) for an upstream stream.
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
    total_duration_secs: Option<f64>,
) -> bool {
    // If already pre-warmed and cached: check if resume buffer needs to be appended
    if let Some(entry) = FAST_START_CACHE.get(original_url) {
        if let Some(pos_secs) = resume_position_secs {
            if entry.resume_buffer.is_none() {
                let target_resume_size = if entry.head_bytes.len() >= FAST_START_4K_HEAD_SIZE {
                    FAST_START_4K_RESUME_SIZE
                } else {
                    FAST_START_DEFAULT_RESUME_SIZE
                };
                let offset_opt = entry.tail_bytes.as_deref().and_then(|tb| find_mkv_cue_offset(tb, pos_secs))
                    .or_else(|| estimate_resume_byte_offset(pos_secs, total_duration_secs, entry.total_size));
                if let Some(start_byte) = offset_opt {
                    let end_byte = (start_byte + target_resume_size as u64 - 1).min(entry.total_size - 1);
                    let resume_range = format!("bytes={}-{}", start_byte, end_byte);
                    let mut res_req = PREWARM_CLIENT.get(&entry.direct_cdn_url);
                    for (k, v) in request_headers {
                        res_req = res_req.header(k.as_str(), v.as_str());
                    }
                    res_req = res_req.header(header::RANGE, resume_range);
                    if let Ok(resp) = res_req.send().await {
                        if resp.status() == StatusCode::PARTIAL_CONTENT {
                            if let Ok(b) = resp.bytes().await {
                                let mut updated = entry.clone();
                                updated.resume_buffer = Some((start_byte, b));
                                FAST_START_CACHE.insert(original_url.to_string(), updated);
                                info!(
                                    %original_url,
                                    item = ?item_id,
                                    start_byte,
                                    end_byte,
                                    "[FAST-START] Appended resume buffer to existing cached entry"
                                );
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
            (q_guard, Some(permit))
        }
        PrewarmPriority::High => (QueuedPrewarmGuard(None), None),
    };

    // Re-check cache before acquiring active lock (in case a High-priority task finished while Low was queued)
    if FAST_START_CACHE.get(original_url).is_some() {
        return true;
    }

    // Coalesce concurrent active downloads for the same stream URL
    let _guard = PREWARM_LOCKS.lock(original_url.to_string()).await;

    // Double check cache after acquiring active lock
    if FAST_START_CACHE.get(original_url).is_some() {
        return true;
    }

    METRICS.prewarms_started.fetch_add(1, Ordering::Relaxed);
    match priority {
        PrewarmPriority::High => METRICS.prewarms_high_started.fetch_add(1, Ordering::Relaxed),
        PrewarmPriority::Low => METRICS.prewarms_low_started.fetch_add(1, Ordering::Relaxed),
    };
    let start_time = Instant::now();

    let target_head_size = head_size.unwrap_or_else(|| calculate_fast_start_head_size(original_url, None, None));
    let target_resume_size = if target_head_size >= FAST_START_4K_HEAD_SIZE {
        FAST_START_4K_RESUME_SIZE
    } else {
        FAST_START_DEFAULT_RESUME_SIZE
    };

    info!(
        %original_url,
        item = ?item_id,
        ?priority,
        head_size_mb = target_head_size / (1024 * 1024),
        resume_size_mb = target_resume_size / (1024 * 1024),
        ?resume_position_secs,
        "[FAST-START] Starting pre-warm download"
    );

    // Step 1: Request the first N MB (16 MB for standard, 32 MB for 4K / large)
    let head_range = format!("bytes=0-{}", target_head_size - 1);
    let mut head_req = PREWARM_CLIENT.get(original_url);
    for (k, v) in request_headers {
        head_req = head_req.header(k.as_str(), v.as_str());
    }
    head_req = head_req.header(header::RANGE, head_range);

    let head_resp = match head_req.send().await {
        Ok(r) => r,
        Err(e) => {
            METRICS.prewarms_failed.fetch_add(1, Ordering::Relaxed);
            warn!(%original_url, error = %e, "[FAST-START] Pre-warm head request failed");
            return false;
        }
    };

    let status = head_resp.status();
    let is_partial = status == StatusCode::PARTIAL_CONTENT;
    let is_ok = status == StatusCode::OK;

    if !is_partial && !is_ok {
        METRICS.prewarms_failed.fetch_add(1, Ordering::Relaxed);
        warn!(%original_url, ?status, "[FAST-START] Pre-warm head request returned non-success status");
        return false;
    }

    // Stream responded with 200/206: populate STREAM_LIVENESS_CACHE immediately so liveness checks are instant
    crate::stream::STREAM_LIVENESS_CACHE.insert(
        original_url.to_string(),
        crate::stream::LivenessEntry::new(true),
    );

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

    // Steps 2 & 3: Read head body AND fetch tail cues + resume chunk in parallel!
    let head_fut = head_resp.bytes();

    let tail_and_resume_fut = async {
        // Step 3: Fetch the 2 MB tail index (MKV Cues)
        let tail_bytes = if total_size > FAST_START_TAIL_SIZE as u64 {
            let tail_start = total_size - FAST_START_TAIL_SIZE as u64;
            let tail_range = format!("bytes={}-{}", tail_start, total_size - 1);
            let mut tail_req = PREWARM_CLIENT.get(&direct_cdn_url);
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

        // Step 4: If resume position requested, find resume byte offset from Cues & fetch resume chunk!
        let resume_buffer = if let Some(pos_secs) = resume_position_secs {
            let offset_opt = tail_bytes
                .as_deref()
                .and_then(|tb| find_mkv_cue_offset(tb, pos_secs))
                .or_else(|| estimate_resume_byte_offset(pos_secs, total_duration_secs, total_size));

            if let Some(start_byte) = offset_opt {
                let end_byte = (start_byte + target_resume_size as u64 - 1).min(total_size - 1);
                let resume_range = format!("bytes={}-{}", start_byte, end_byte);
                let mut res_req = PREWARM_CLIENT.get(&direct_cdn_url);
                for (k, v) in request_headers {
                    res_req = res_req.header(k.as_str(), v.as_str());
                }
                res_req = res_req.header(header::RANGE, resume_range);

                match res_req.send().await {
                    Ok(r) if r.status() == StatusCode::PARTIAL_CONTENT => {
                        match r.bytes().await {
                            Ok(b) => {
                                info!(
                                    %original_url,
                                    start_byte,
                                    end_byte,
                                    resume_size_bytes = b.len(),
                                    "[FAST-START] Successfully pre-warmed resume chunk into RAM"
                                );
                                Some((start_byte, b))
                            }
                            Err(e) => {
                                warn!(%original_url, error = %e, "[FAST-START] Failed reading resume chunk body");
                                None
                            }
                        }
                    }
                    Ok(r) => {
                        warn!(%original_url, status = ?r.status(), "[FAST-START] Resume fetch returned non-206 status");
                        None
                    }
                    Err(e) => {
                        warn!(%original_url, error = %e, "[FAST-START] Resume fetch request error");
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        (tail_bytes, resume_buffer)
    };

    let (head_res, (tail_bytes, resume_buffer)) = tokio::join!(head_fut, tail_and_resume_fut);

    let head_bytes = match head_res {
        Ok(b) => b,
        Err(e) => {
            METRICS.prewarms_failed.fetch_add(1, Ordering::Relaxed);
            warn!(%original_url, error = %e, "[FAST-START] Failed reading head response body");
            return false;
        }
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
                let _ = axum::serve(listener, app).await;
            });
            Ok::<u16, std::io::Error>(port)
        })
        .await
        .ok()
        .copied()
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
    if !prewarm_fast_start(None, url, &HashMap::new(), PrewarmPriority::High, None, None, None).await {
        return url.to_string();
    }
    let Some(port) = ensure_local_probe_server().await else {
        return url.to_string();
    };
    let encoded: String = url::form_urlencoded::byte_serialize(url.as_bytes()).collect();
    info!(
        %url,
        local_port = port,
        "[FAST-START] Routing ffprobe through local RAM buffer"
    );
    format!("http://127.0.0.1:{port}/fast-start-probe?url={encoded}")
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
    if is_browser_video_request(headers) {
        if let Some(entry) = FAST_START_CACHE.get(url) {
            info!(
                %item_id,
                direct_cdn = %entry.direct_cdn_url,
                "[FAST-START] Browser <video> request detected; issuing consistent 307 to direct CDN from byte 0"
            );
            let mut redirect = Redirect::temporary(&entry.direct_cdn_url).into_response();
            redirect.headers_mut().insert(
                "X-Remux-FastStart",
                HeaderValue::from_static("browser-direct-cdn"),
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

    // 2. If not cached, await in-flight prewarm up to 4000ms for this exact URL
    if entry.is_none() {
        if PREWARM_LOCKS.contains_key(&url.to_string()) {
            info!(%item_id, %url, "[FAST-START] Pre-warm currently in flight; awaiting completion up to 4000ms");
            let _ = tokio::time::timeout(
                Duration::from_millis(4000),
                PREWARM_LOCKS.lock_if_exists(&url.to_string()),
            )
            .await;

            // Re-check cache after lock release
            entry = FAST_START_CACHE.get(url);

            if entry.is_some() {
                info!(%item_id, "[FAST-START] In-flight pre-warm successfully coalesced into cache!");
            }
        } else {
            // Uncached and nothing in-flight: don't block this request.
            // Spawn pre-warm in background for this exact URL so subsequent seeks benefit, but immediately fall through to 307 redirect.
            info!(%item_id, %url, "[FAST-START] Uncached & not in-flight: spawning background pre-warm and immediately falling back to direct redirect");
            let url_str = url.to_string();
            let headers_clone = upstream_headers.cloned().unwrap_or_default();
            tokio::spawn(async move {
                prewarm_fast_start(Some(item_id), &url_str, &headers_clone, PrewarmPriority::High, None, None, None).await;
            });
        }
    }

    let entry = entry?;

    let range_header = headers.get(header::RANGE).and_then(|v| v.to_str().ok());

    match range_header {
        // Case 1: Initial probe / head sub-range request (any Range starting inside the cached head buffer)
        Some(rng) if is_head_range(rng, entry.head_bytes.len(), entry.total_size) => {
            METRICS.hits_head.fetch_add(1, Ordering::Relaxed);
            let head_len = entry.head_bytes.len();
            let (req_start, req_end) =
                parse_range_bounds(rng, entry.total_size).unwrap_or((0, entry.total_size.saturating_sub(1)));

            let start_idx = req_start as usize;
            let end_idx = (req_end as usize).min(head_len.saturating_sub(1));
            let slice = entry.head_bytes.slice(start_idx..=end_idx);
            let slice_len = slice.len();
            let end_offset = req_start + (slice_len as u64).saturating_sub(1);

            METRICS.bytes_served_lan.fetch_add(slice_len as u64, Ordering::Relaxed);

            info!(
                %item_id,
                req_start,
                end_offset,
                slice_len,
                total_size = entry.total_size,
                "[FAST-START] Buffer hit (head): serving {} bytes locally over LAN",
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
            METRICS.hits_head.fetch_add(1, Ordering::Relaxed);
            let head_len = entry.head_bytes.len();
            METRICS.bytes_served_lan.fetch_add(head_len as u64, Ordering::Relaxed);

            info!(
                %item_id,
                head_len,
                total_size = entry.total_size,
                "[FAST-START] Buffer hit (no range): serving head bytes locally over LAN"
            );

            let resp = Response::builder()
                .status(StatusCode::PARTIAL_CONTENT)
                .header(header::CONTENT_TYPE, &entry.content_type)
                .header(
                    header::CONTENT_RANGE,
                    format!("bytes 0-{}/{}", head_len.saturating_sub(1), entry.total_size),
                )
                .header(header::CONTENT_LENGTH, head_len.to_string())
                .header(header::ACCEPT_RANGES, "bytes")
                .header("X-Remux-FastStart", "hit-head")
                .body(Body::from(entry.head_bytes.clone()))
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

            // Fall through to direct CDN redirect if tail is missing or sub-range not covered
            METRICS.redirects_issued.fetch_add(1, Ordering::Relaxed);
            info!(
                %item_id,
                range = %rng,
                direct_cdn = %entry.direct_cdn_url,
                "[FAST-START] Redirecting tail request directly to Debrid CDN (307)"
            );
            let mut redirect = Redirect::temporary(&entry.direct_cdn_url).into_response();
            redirect.headers_mut().insert("X-Remux-FastStart", HeaderValue::from_static("redirect-direct-cdn"));
            Some(redirect)
        }

        // Case 4: Resume buffer hit (when resuming playback from the middle)
        Some(rng) if is_resume_range(rng, entry.resume_buffer.as_ref(), entry.total_size) => {
            if let Some((resume_start, ref resume_bytes)) = entry.resume_buffer {
                let (req_start, req_end) = parse_range_bounds(rng, entry.total_size)
                    .unwrap_or((resume_start, entry.total_size.saturating_sub(1)));
                let resume_len = resume_bytes.len() as u64;
                let resume_end = resume_start + resume_len;
                if req_start >= resume_start && req_start < resume_end {
                    let offset = (req_start - resume_start) as usize;
                    let end_in_resume = (req_end - resume_start) as usize;
                    let end_clamped = end_in_resume.min(resume_bytes.len() - 1);
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
                        "[FAST-START] Buffer hit (resume): serving {} bytes locally over LAN",
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

            // Fall through to direct CDN redirect if resume chunk doesn't cover
            METRICS.redirects_issued.fetch_add(1, Ordering::Relaxed);
            info!(
                %item_id,
                range = %rng,
                direct_cdn = %entry.direct_cdn_url,
                "[FAST-START] Resume range miss: redirecting to Debrid CDN (307)"
            );
            let mut redirect = Redirect::temporary(&entry.direct_cdn_url).into_response();
            redirect.headers_mut().insert("X-Remux-FastStart", HeaderValue::from_static("redirect-direct-cdn"));
            Some(redirect)
        }

        // Case 5: Video cluster stream / seek / continuous playback
        // Issue 307 Temporary Redirect DIRECTLY to Debrid CDN edge
        Some(rng) => {
            METRICS.redirects_issued.fetch_add(1, Ordering::Relaxed);
            info!(
                %item_id,
                range = %rng,
                direct_cdn = %entry.direct_cdn_url,
                "[FAST-START] Direct CDN handoff: issuing 307 directly to CDN edge (0 Mbps server upload)"
            );
            let mut redirect = Redirect::temporary(&entry.direct_cdn_url).into_response();
            redirect.headers_mut().insert("X-Remux-FastStart", HeaderValue::from_static("redirect-direct-cdn"));
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

/// Extracts the cluster byte offset for a given timestamp (in seconds) from an MKV tail index (Cues).
pub fn find_mkv_cue_offset(tail_bytes: &[u8], target_time_secs: f64) -> Option<u64> {
    if tail_bytes.len() < 16 {
        return None;
    }
    // Search for Cues element ID: [0x1C, 0x53, 0xBB, 0x6B]
    let cues_id = [0x1Cu8, 0x53, 0xBB, 0x6B];
    let cues_pos = tail_bytes.windows(4).position(|w| w == cues_id)?;

    let mut cursor = cues_pos + 4;
    let (_cues_len, vint_len) = read_ebml_vint(&tail_bytes[cursor..])?;
    cursor += vint_len;

    let target_time_ms = (target_time_secs * 1000.0) as u64;
    let mut best_cue: Option<(u64, u64)> = None; // (time_ms, cluster_offset)

    // Iterate through CuePoint elements (ID: 0xBB)
    while cursor < tail_bytes.len() {
        if tail_bytes[cursor] != 0xBB {
            cursor += 1;
            continue;
        }
        cursor += 1; // skip 0xBB
        let Some((point_len, point_vint_len)) = read_ebml_vint(&tail_bytes[cursor..]) else {
            break;
        };
        cursor += point_vint_len;
        let point_end = (cursor + point_len as usize).min(tail_bytes.len());

        let mut sub_cursor = cursor;
        let mut cue_time_ms: Option<u64> = None;
        let mut cue_cluster_pos: Option<u64> = None;

        while sub_cursor < point_end {
            let elem_id = tail_bytes[sub_cursor];
            sub_cursor += 1;
            let Some((elem_len, elem_vint_len)) = read_ebml_vint(&tail_bytes[sub_cursor..]) else {
                break;
            };
            sub_cursor += elem_vint_len;
            let elem_end = (sub_cursor + elem_len as usize).min(point_end);

            if elem_id == 0xB3 {
                // CueTime
                cue_time_ms = Some(read_ebml_uint(&tail_bytes[sub_cursor..elem_end]));
            } else if elem_id == 0xB7 {
                // CueTrackPositions: scan inside for CueClusterPosition (0xF1)
                let mut track_cursor = sub_cursor;
                while track_cursor < elem_end {
                    let tid = tail_bytes[track_cursor];
                    track_cursor += 1;
                    let Some((tlen, tvint_len)) = read_ebml_vint(&tail_bytes[track_cursor..]) else {
                        break;
                    };
                    track_cursor += tvint_len;
                    let tend = (track_cursor + tlen as usize).min(elem_end);
                    if tid == 0xF1 {
                        cue_cluster_pos = Some(read_ebml_uint(&tail_bytes[track_cursor..tend]));
                    }
                    track_cursor = tend;
                }
            }
            sub_cursor = elem_end;
        }

        if let (Some(t), Some(c)) = (cue_time_ms, cue_cluster_pos) {
            if t <= target_time_ms {
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

    // In Matroska, CueClusterPosition is relative to Segment data offset (~64 bytes into file).
    best_cue.map(|(_, pos)| pos + 64)
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
    let mask = 0xFF >> num_bytes;
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

/// Fallback linear estimation for non-MKV files or when Cues are absent from the tail.
fn estimate_resume_byte_offset(
    pos_secs: f64,
    duration_secs: Option<f64>,
    total_size: u64,
) -> Option<u64> {
    let duration = duration_secs.filter(|d| *d > 0.0)?;
    if pos_secs <= 0.0 || pos_secs >= duration {
        return None;
    }
    let ratio = pos_secs / duration;
    let byte_offset = (ratio * total_size as f64) as u64;
    Some(byte_offset.min(total_size.saturating_sub(1)))
}

/// Looks up any saved playback position for a user and item.
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
            "resume_buffer_offset": entry.resume_buffer.as_ref().map(|(o, _)| *o),
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
            "max_items": FAST_START_MAX_ITEMS,
            "cached_items_count": cached_items.len(),
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

    #[tokio::test]
    async fn head_subrange_requests_served_from_ram() {
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
                total_size: 1_000_000,
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
            "bytes 8154-16383/1000000"
        );
        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&body_bytes[..], &head_data[8154..16384]);
    }

    #[tokio::test]
    async fn resume_buffer_range_served_from_ram() {
        let test_url = "http://127.0.0.1:9999/resume-test.mkv";
        let resume_start: u64 = 500_000_000;
        let resume_data: Vec<u8> = (0..32768u32).map(|i| (i % 251) as u8).collect();
        FAST_START_CACHE.insert(
            test_url.to_string(),
            FastStartEntry {
                item_id: Some(Uuid::nil()),
                original_url: test_url.to_string(),
                direct_cdn_url: "http://cdn.example.com/resume.mkv".to_string(),
                head_bytes: Bytes::from(vec![0xAA; 1024]),
                tail_bytes: None,
                resume_buffer: Some((resume_start, Bytes::from(resume_data.clone()))),
                total_size: 1_000_000_000,
                content_type: "video/x-matroska".to_string(),
                created_at: Instant::now(),
            },
        );

        // Infuse seeks directly to byte 500,001,000 (inside the resume buffer)
        let mut headers = HeaderMap::new();
        headers.insert(header::RANGE, HeaderValue::from_static("bytes=500001000-"));
        let resp = handle_playback_request(Uuid::nil(), test_url, &headers, None)
            .await
            .expect("should hit fast-start cache at resume position");

        assert_eq!(resp.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(
            resp.headers().get("X-Remux-FastStart").unwrap(),
            "hit-resume"
        );
        assert_eq!(
            resp.headers().get(header::CONTENT_RANGE).unwrap(),
            "bytes 500001000-500032767/1000000000"
        );
        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&body_bytes[..], &resume_data[1000..32768]);
    }

    #[tokio::test]
    async fn high_priority_prewarm_overtakes_queued_low_priority_on_same_url() {
        // Exhaust all 2 permits of the background semaphore
        let p1 = FAST_START_PREWARM_SEMAPHORE.acquire().await.unwrap();
        let p2 = FAST_START_PREWARM_SEMAPHORE.acquire().await.unwrap();

        let server = axum::Router::new().route(
            "/video.mkv",
            axum::routing::get(|| async {
                Response::builder()
                    .status(StatusCode::PARTIAL_CONTENT)
                    .header(header::CONTENT_TYPE, "video/x-matroska")
                    .header(header::CONTENT_RANGE, "bytes 0-15/16")
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
            prewarm_fast_start(None, &url_low, &HashMap::new(), PrewarmPriority::Low, None, None, None).await
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
            prewarm_fast_start(None, &test_url, &HashMap::new(), PrewarmPriority::High, None, None, None),
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
}
