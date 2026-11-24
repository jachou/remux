use axum::{
    extract::{Path, State},
    response::IntoResponse,
};
use remux_macros::get;
use uuid::Uuid;

use crate::{OptionExt, ResultExt};
use axum_anyhow::ApiResult as Result;

use crate::{AppState, db, stream::StreamDescriptor};

/// Returns `true` only for direct local loopback requests (such as internal
/// `ffmpeg`/`ffprobe` reading `http://127.0.0.1:{port}/stream/{id}`), never
/// for requests forwarded through Cloudflare Tunnel or any reverse proxy.
fn is_direct_loopback_request(headers: &axum::http::HeaderMap) -> bool {
    const PROXY_HEADERS: [&str; 6] = [
        "cf-connecting-ip",
        "cf-ray",
        "x-forwarded-for",
        "x-real-ip",
        "forwarded",
        "true-client-ip",
    ];
    if PROXY_HEADERS
        .iter()
        .any(|h| headers.contains_key(*h))
    {
        return false;
    }
    let Some(host) = headers
        .get(axum::http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
    else {
        return false;
    };
    let host_no_port = if let Some(rest) = host.strip_prefix('[') {
        rest.split(']')
            .next()
            .unwrap_or("")
    } else {
        host.split(':')
            .next()
            .unwrap_or("")
    };
    matches!(host_no_port, "127.0.0.1" | "::1" | "localhost")
}

async fn is_authenticated_stream_caller(
    state: &AppState,
    jfauth: &crate::db::auth::JellyfinAuthHeader,
) -> bool {
    let Some(token) = jfauth
        .token
        .as_deref()
    else {
        return false;
    };
    let db = &state
        .ctx
        .db;
    if let Ok(Some(device)) = crate::db::auth::Device::get_by_access_token(db, token).await {
        if let Ok(Some(user)) = db::User::get_by_id(db, &device.user_id).await {
            return !user.is_disabled();
        }
        return false;
    }
    matches!(db::ApiKey::get_by_token(db, token).await, Ok(Some(_)))
}

/// Proxy any stream stored in `db::Media.stream_info` to the caller.
///
/// Handles all URL schemes transparently via [`crate::stream::StreamSource`].
/// Addon-owned streams (`Opendal`) are dispatched to the addon's `serve_stream`.
/// Requires authentication except for direct local loopback requests from
/// internal FFmpeg/FFprobe processes.
#[get("/stream/{id}")]
pub async fn stream_proxy(
    headers: axum::http::HeaderMap,
    State(state): State<AppState>,
    jfauth: crate::db::auth::JellyfinAuthHeader,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse> {
    if !is_authenticated_stream_caller(&state, &jfauth).await
        && !is_direct_loopback_request(&headers)
    {
        return Err(anyhow::anyhow!("unauthorized"))
            .context_unauthorized("Authentication required");
    }

    let media = db::Media::get_by_id(
        &state
            .ctx
            .db,
        &id,
    )
    .await?
    .context_not_found("not found")?;

    let descriptor = media
        .stream_info
        .map(|si| si.descriptor)
        .context_not_found("media has no URL")?;

    if let Some(addon_id) = descriptor.addon_id() {
        let addon = state
            .ctx
            .addons
            .get(addon_id)
            .context_not_found("addon not found")?;
        let stream = addon
            .stream
            .as_ref()
            .context_not_found("addon does not support streams")?;
        return stream
            .serve_stream(&descriptor, &headers)
            .await;
    }

    if matches!(descriptor, StreamDescriptor::Torrent { .. }) {
        let cfg = db::Settings::get_config_or_default(
            &state
                .ctx
                .db,
        )
        .await;
        if !cfg
            .p2p_enabled
            .unwrap_or(true)
        {
            return Err(anyhow::anyhow!("P2P disabled")).context_bad_request(
                "P2P streams are disabled by the server administrator",
            );
        }
    }

    descriptor
        .into_source()
        .serve(&state, &headers)
        .await
}

#[cfg(test)]
mod tests {
    use http::{HeaderMap, HeaderValue, StatusCode};

    use super::is_direct_loopback_request;
    use crate::integration_test::authenticated_server;

    #[test]
    fn loopback_helper_rejects_forwarded_requests() {
        let mut h = HeaderMap::new();
        h.insert(http::header::HOST, HeaderValue::from_static("127.0.0.1:3000"));
        assert!(is_direct_loopback_request(&h));

        h.insert("cf-connecting-ip", HeaderValue::from_static("203.0.113.5"));
        assert!(!is_direct_loopback_request(&h));

        let mut ext = HeaderMap::new();
        ext.insert(http::header::HOST, HeaderValue::from_static("remux.example.com"));
        assert!(!is_direct_loopback_request(&ext));
    }

    #[tokio::test]
    async fn stream_proxy_blocks_unauthenticated_remote_and_allows_direct_loopback() {
        let (server, _guard, _token) = authenticated_server().await;
        let id = uuid::Uuid::new_v4();

        // Proxied request without auth must return 401.
        server
            .get(&format!("/stream/{id}"))
            .add_header("host", HeaderValue::from_static("remux.example.com"))
            .add_header("cf-connecting-ip", HeaderValue::from_static("203.0.113.9"))
            .expect_failure()
            .await
            .assert_status(StatusCode::UNAUTHORIZED);

        // Direct loopback request (internal ffmpeg/ffprobe) passes auth and hits 404.
        server
            .get(&format!("/stream/{id}"))
            .add_header("host", HeaderValue::from_static("127.0.0.1:3000"))
            .expect_failure()
            .await
            .assert_status(StatusCode::NOT_FOUND);
    }
}
