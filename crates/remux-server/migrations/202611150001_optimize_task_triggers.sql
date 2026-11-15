-- Remove redundant 5-hour RefreshAllMeta trigger (RefreshLibrary already refreshes outdated metadata incrementally)
DELETE FROM task_triggers
WHERE lower(task_id) = 'refreshallmeta' AND cron = '0 30 */5 * * *';

-- Offset RefreshPopularity daily trigger from 10:00:00 UTC to 10:17:00 UTC to avoid
-- colliding with RefreshLibrary at 10:00:00 and the global top-of-the-hour RemuxDB spike
UPDATE task_triggers
SET cron = '0 17 10 * * *'
WHERE id = 'default-refreshpopularity-daily' AND cron = '0 0 10 * * *';
