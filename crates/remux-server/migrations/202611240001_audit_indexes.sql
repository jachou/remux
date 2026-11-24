-- Partial index for fast-start background probe upgrades looking up stream rows by HTTP URL.
CREATE INDEX IF NOT EXISTS idx_media_stream_http_url
    ON media(json_extract(stream_info, '$.descriptor.Http.url'))
    WHERE kind = 'stream';

-- Composite (kind, id) index for O(N) cursor pagination in Media::get_refreshable / RefreshAllMetaTask.
CREATE INDEX IF NOT EXISTS idx_media_kind_id
    ON media(kind, id);
