-- /items/latest commonly filters by a single IncludeItemTypes (e.g. Episode)
-- and sorts by DateCreated (`datetime(created_at)` DESC). Without `kind` leading
-- the expression index, SQLite picks `idx_media_kind_enabled` and sorts all
-- ~600k episode rows in a temporary B-tree.
CREATE INDEX IF NOT EXISTS idx_media_kind_created_at_sort
    ON media(kind, datetime(created_at));
