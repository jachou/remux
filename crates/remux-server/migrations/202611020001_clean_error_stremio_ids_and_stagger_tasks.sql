-- Clean up any error stubs inadvertently saved into media.external_ids
UPDATE media
SET external_ids = json_remove(external_ids, '$.custom_stremio_id')
WHERE json_extract(external_ids, '$.custom_stremio_id') LIKE '%error%'
   OR json_extract(external_ids, '$.custom_stremio_id') LIKE 'aiostreamserror.%';

-- Stagger RefreshAllMeta by 30 minutes so it does not collide with RefreshLibrary at :00
UPDATE task_triggers
SET cron = '0 30 */5 * * *'
WHERE task_id = 'refreshallmeta' AND cron = '0 0 */5 * * *';
