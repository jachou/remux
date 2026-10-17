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

/// Head buffer size in bytes: 16 MB.
pub const FAST_START_HEAD_SIZE: usize = 16 * 1024 * 1024;

/// Tail buffer size for MKV Cues / seekhead index: 128 KB.
pub const FAST_START_TAIL_SIZE: usize = 128 * 1024;

/// Maximum number of items held in memory simultaneously.
pub const FAST_START_MAX_ITEMS: usize = 15;

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
    pub created_at: Instant,
}

/// Global metrics for fast-start acceleration.
pub struct FastStartMetrics {
    pub hits_head: AtomicU64,
    pub hits_tail: AtomicU64,
    pub redirects_issued: AtomicU64,
    pub misses: AtomicU64,
    pub prewarms_started: AtomicU64,
    pub prewarms_succeeded: AtomicU64,
    pub prewarms_failed: AtomicU64,
    pub bytes_served_lan: AtomicU64,
}

impl FastStartMetrics {
    const fn new() -> Self {
        Self {
            hits_head: AtomicU64::new(0),
            hits_tail: AtomicU64::new(0),
            redirects_issued: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            prewarms_started: AtomicU64::new(0),
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

/// Lookup mapping from Jellyfin Media/Item UUID to original stream URL.
pub(crate) static ITEM_TO_STREAM_URL: LazyLock<moka::sync::Cache<Uuid, String>> =
    LazyLock::new(|| {
        moka::sync::Cache::builder()
            .max_capacity(100)
            .time_to_live(FAST_START_TTL)
            .build()
    });

/// Keyed lock to prevent duplicate concurrent pre-warms for the same URL.
static PREWARM_LOCKS: LazyLock<crate::keyed_lock::KeyedLock<String>> =
    LazyLock::new(crate::keyed_lock::KeyedLock::new);

/// Dedicated reqwest client for pre-warming (follows HTTP 302/307 redirects to CDN).
static PREWARM_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .expect("failed to build fast-start prewarm client")
});

/// Pre-warms the head buffer (64 MB) and tail index (128 KB) for an upstream stream.
///
/// Runs asynchronously in the background. Follows upstream redirects (e.g. Stremthru -> Torbox CDN),
/// resolves the direct CDN edge URL, and caches the bytes in memory.
pub async fn prewarm_fast_start(
    item_id: Option<Uuid>,
    original_url: &str,
    request_headers: &HashMap<String, String>,
) -> bool {
    // Record item -> stream mapping right away so concurrent requests can find it
    if let Some(id) = item_id {
        ITEM_TO_STREAM_URL.insert(id, original_url.to_string());
    }

    // If already pre-warmed and cached, skip!
    if let Some(entry) = FAST_START_CACHE.get(original_url) {
        if let Some(id) = item_id {
            ITEM_TO_STREAM_URL.insert(id, original_url.to_string());
        }
        debug!(
            %original_url,
            item = ?item_id,
            head_bytes = entry.head_bytes.len(),
            "[FAST-START] Stream already cached; skipping prewarm"
        );
        return true;
    }

    // Coalesce concurrent prewarms for the same stream URL
    let _guard = PREWARM_LOCKS.lock(original_url.to_string()).await;

    // Double check cache after acquiring lock
    if FAST_START_CACHE.get(original_url).is_some() {
        if let Some(id) = item_id {
            ITEM_TO_STREAM_URL.insert(id, original_url.to_string());
        }
        return true;
    }

    METRICS.prewarms_started.fetch_add(1, Ordering::Relaxed);
    let start_time = Instant::now();

    info!(
        %original_url,
        item = ?item_id,
        target_size_mb = FAST_START_HEAD_SIZE / (1024 * 1024),
        "[FAST-START] Starting background pre-warm"
    );

    // Step 1: Request the first 64 MB (bytes 0 to 67108863)
    let head_range = format!("bytes=0-{}", FAST_START_HEAD_SIZE - 1);
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
    if !status.is_success() && status != StatusCode::PARTIAL_CONTENT {
        METRICS.prewarms_failed.fetch_add(1, Ordering::Relaxed);
        warn!(%original_url, ?status, "[FAST-START] Pre-warm head request returned non-success status");
        return false;
    }

    // Capture the direct CDN edge URL after Stremthru redirect
    let direct_cdn_url = head_resp.url().as_str().to_string();

    // Parse total file size from Content-Range header: bytes 0-67108863/4694973563
    let total_size = head_resp
        .headers()
        .get(header::CONTENT_RANGE)
        .and_then(|v| v.to_str().ok())
        .and_then(|cr| cr.split('/').last())
        .and_then(|s| s.parse::<u64>().ok())
        .or_else(|| head_resp.content_length());

    let total_size = match total_size {
        Some(s) => s,
        None => {
            METRICS.prewarms_failed.fetch_add(1, Ordering::Relaxed);
            warn!(%original_url, "[FAST-START] Could not determine total file size for prewarm");
            return false;
        }
    };

    // Read the head bytes
    let head_bytes = match head_resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            METRICS.prewarms_failed.fetch_add(1, Ordering::Relaxed);
            warn!(%original_url, error = %e, "[FAST-START] Failed reading head response body");
            return false;
        }
    };

    // Step 2: Fetch the tail index (last 128 KB for MKV Cues), if file is larger than head
    let tail_bytes = if total_size > FAST_START_TAIL_SIZE as u64 {
        let tail_start = total_size - FAST_START_TAIL_SIZE as u64;
        let tail_range = format!("bytes={}-{}", tail_start, total_size - 1);
        let mut tail_req = PREWARM_CLIENT.get(&direct_cdn_url);
        for (k, v) in request_headers {
            tail_req = tail_req.header(k.as_str(), v.as_str());
        }
        tail_req = tail_req.header(header::RANGE, tail_range);

        match tail_req.send().await {
            Ok(r) if r.status().is_success() || r.status() == StatusCode::PARTIAL_CONTENT => {
                r.bytes().await.ok()
            }
            Ok(r) => {
                warn!(%original_url, status = ?r.status(), "[FAST-START] Tail fetch returned non-200/206 status");
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

    let entry = FastStartEntry {
        item_id,
        original_url: original_url.to_string(),
        direct_cdn_url: direct_cdn_url.clone(),
        total_size,
        head_bytes: head_bytes.clone(),
        tail_bytes: tail_bytes.clone(),
        created_at: Instant::now(),
    };

    FAST_START_CACHE.insert(original_url.to_string(), entry);
    if let Some(id) = item_id {
        ITEM_TO_STREAM_URL.insert(id, original_url.to_string());
    }

    METRICS.prewarms_succeeded.fetch_add(1, Ordering::Relaxed);
    let elapsed = start_time.elapsed();

    info!(
        %original_url,
        item = ?item_id,
        direct_cdn_host = head_resp_host(&direct_cdn_url),
        head_size_bytes = head_bytes.len(),
        tail_size_bytes = tail_bytes.as_ref().map(|b| b.len()).unwrap_or(0),
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

/// Evaluates an incoming stream playback request from a media player (e.g. Infuse)
/// and handles it via local fast-start cache or direct CDN handoff.
///
/// Returns `Some(Response)` if handled, or `None` if the stream is not cached.
pub async fn handle_playback_request(
    item_id: Uuid,
    url: &str,
    headers: &HeaderMap,
) -> Option<Response> {
    // 1. Direct lookup by URL, or fallback to Item ID mapping
    let mut entry = FAST_START_CACHE
        .get(url)
        .or_else(|| ITEM_TO_STREAM_URL.get(&item_id).and_then(|u| FAST_START_CACHE.get(&u)));

    // 2. If not cached, check if background pre-warm is currently in flight for this stream
    if entry.is_none() {
        let candidate_url = if PREWARM_LOCKS.contains_key(&url.to_string()) {
            Some(url.to_string())
        } else {
            ITEM_TO_STREAM_URL.get(&item_id).filter(|u| PREWARM_LOCKS.contains_key(u))
        };

        if let Some(c_url) = candidate_url {
            info!(%item_id, %c_url, "[FAST-START] Pre-warm currently in flight; awaiting completion up to 3000ms");
            let _ = tokio::time::timeout(
                Duration::from_millis(3000),
                PREWARM_LOCKS.lock_if_exists(&c_url),
            )
            .await;

            // Re-check cache after lock release
            entry = FAST_START_CACHE
                .get(url)
                .or_else(|| ITEM_TO_STREAM_URL.get(&item_id).and_then(|u| FAST_START_CACHE.get(&u)));

            if entry.is_some() {
                info!(%item_id, "[FAST-START] In-flight pre-warm successfully coalesced into cache!");
            }
        }
    }

    let entry = entry?;

    let range_header = headers.get(header::RANGE).and_then(|v| v.to_str().ok());

    match range_header {
        // Case 1: Initial probe / head request (Range: bytes=0- or bytes=0-X)
        Some(rng) if rng.starts_with("bytes=0-") || rng == "bytes=0-" => {
            METRICS.hits_head.fetch_add(1, Ordering::Relaxed);
            let head_len = entry.head_bytes.len();

            // Check if client requested a bounded sub-range of the head
            let requested_end = rng[6..]
                .split('-')
                .nth(1)
                .and_then(|s| s.parse::<usize>().ok());

            let (slice, slice_len, end_offset) = match requested_end {
                Some(end) if end < head_len => {
                    let s = entry.head_bytes.slice(0..=end);
                    let len = s.len();
                    (s, len, end as u64)
                }
                _ => (entry.head_bytes.clone(), head_len, head_len.saturating_sub(1) as u64),
            };

            METRICS.bytes_served_lan.fetch_add(slice_len as u64, Ordering::Relaxed);

            info!(
                %item_id,
                slice_len,
                total_size = entry.total_size,
                "[FAST-START] Buffer hit (head): serving {} bytes locally over LAN",
                slice_len
            );

            let resp = Response::builder()
                .status(StatusCode::PARTIAL_CONTENT)
                .header(header::CONTENT_TYPE, "video/x-matroska")
                .header(
                    header::CONTENT_RANGE,
                    format!("bytes 0-{}/{}", end_offset, entry.total_size),
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
                .header(header::CONTENT_TYPE, "video/x-matroska")
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
                let req_start = parse_range_start(rng).unwrap_or(tail_start);
                let offset = req_start.saturating_sub(tail_start) as usize;

                if offset < tail.len() {
                    let slice = tail.slice(offset..);
                    let slice_len = slice.len();
                    METRICS.hits_tail.fetch_add(1, Ordering::Relaxed);
                    METRICS.bytes_served_lan.fetch_add(slice_len as u64, Ordering::Relaxed);

                    info!(
                        %item_id,
                        req_start,
                        slice_len,
                        total_size = entry.total_size,
                        "[FAST-START] Buffer hit (tail/cues): serving {} bytes locally over LAN",
                        slice_len
                    );

                    let resp = Response::builder()
                        .status(StatusCode::PARTIAL_CONTENT)
                        .header(header::CONTENT_TYPE, "video/x-matroska")
                        .header(
                            header::CONTENT_RANGE,
                            format!("bytes {}-{}/{}", req_start, entry.total_size - 1, entry.total_size),
                        )
                        .header(header::CONTENT_LENGTH, slice_len.to_string())
                        .header(header::ACCEPT_RANGES, "bytes")
                        .header("X-Remux-FastStart", "hit-tail")
                        .body(Body::from(slice))
                        .unwrap();

                    return Some(resp);
                }
            }

            // Fall through to direct CDN redirect if tail is missing
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

        // Case 4: Video cluster stream / seek / continuous playback
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

/// Helper to parse the starting byte from a Range header: "bytes=12345-" -> 12345
fn parse_range_start(range_str: &str) -> Option<u64> {
    if !range_str.starts_with("bytes=") {
        return None;
    }
    let after_prefix = &range_str[6..];
    let start_part = after_prefix.split('-').next()?;
    start_part.parse::<u64>().ok()
}

/// Determines whether a Range header is requesting the tail (MKV Cues / seekhead) region.
fn is_tail_range(range_str: &str, total_size: u64) -> bool {
    let Some(start) = parse_range_start(range_str) else {
        return false;
    };
    let tail_threshold = total_size.saturating_sub(FAST_START_TAIL_SIZE as u64 + 1024);
    start >= tail_threshold
}

/// Returns a JSON representation of all fast-start metrics and cached items.
pub fn stats_json() -> serde_json::Value {
    let mut cached_items = Vec::new();
    let mut total_ram_bytes: u64 = 0;

    for (url, entry) in FAST_START_CACHE.iter() {
        let head_len = entry.head_bytes.len() as u64;
        let tail_len = entry.tail_bytes.as_ref().map(|b| b.len() as u64).unwrap_or(0);
        let item_ram = head_len + tail_len;
        total_ram_bytes += item_ram;

        cached_items.push(json!({
            "item_id": entry.item_id,
            "original_url": entry.original_url,
            "direct_cdn_host": head_resp_host(&entry.direct_cdn_url),
            "total_file_size_bytes": entry.total_size,
            "head_buffer_bytes": head_len,
            "tail_buffer_bytes": tail_len,
            "ram_usage_mb": (item_ram as f64) / (1024.0 * 1024.0),
            "age_seconds": entry.created_at.elapsed().as_secs(),
        }));
    }

    json!({
        "metrics": {
            "hits_head": METRICS.hits_head.load(Ordering::Relaxed),
            "hits_tail": METRICS.hits_tail.load(Ordering::Relaxed),
            "redirects_issued": METRICS.redirects_issued.load(Ordering::Relaxed),
            "misses": METRICS.misses.load(Ordering::Relaxed),
            "prewarms_started": METRICS.prewarms_started.load(Ordering::Relaxed),
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
