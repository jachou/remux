-- Multi-user isolation for Jellyfin display preferences (PRIMARY KEY (user_id, client, id)
-- instead of PRIMARY KEY (id), which caused users/clients to overwrite each other's "usersettings")
-- plus partial indexes for multi-user Continue Watching & Favorites queries.

CREATE TABLE jellyfin_display_prefs_new (
    id      TEXT NOT NULL,
    user_id BLOB NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    client  TEXT NOT NULL DEFAULT '',
    data    TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (user_id, client, id)
);

INSERT OR IGNORE INTO jellyfin_display_prefs_new (id, user_id, client, data)
SELECT id, user_id, COALESCE(client, ''), data
FROM jellyfin_display_prefs;

DROP TABLE jellyfin_display_prefs;
ALTER TABLE jellyfin_display_prefs_new RENAME TO jellyfin_display_prefs;

CREATE INDEX IF NOT EXISTS idx_ums_user_resumable
    ON user_media_state(user_id, media_id)
    WHERE playback_position > 0;

CREATE INDEX IF NOT EXISTS idx_ums_user_favorite
    ON user_media_state(user_id, media_id)
    WHERE favorite = 1;
