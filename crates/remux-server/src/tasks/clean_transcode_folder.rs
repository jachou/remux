use anyhow::Result;
use async_trait::async_trait;
#[cfg(unix)]
use libc;
use std::{collections::HashSet, sync::Arc};
use tracing::{info, warn};

use super::{ProgressReporter, Task, TaskCategory, TaskService};
use crate::AppContext;

pub struct CleanTranscodeFolderTask;

#[async_trait]
impl Task for CleanTranscodeFolderTask {
    fn key(&self) -> &str {
        "CleanTranscodeFolder"
    }
    fn name(&self) -> &str {
        "Clean Transcode Folder"
    }
    fn description(&self) -> &str {
        "Deletes temporary files left over from transcoding sessions."
    }
    fn short_description(&self) -> &str {
        "Deletes leftover temp transcode files"
    }
    fn category(&self) -> TaskCategory {
        TaskCategory::Maintenance
    }

    async fn run(
        &self,
        ctx: AppContext,
        _tasks: Arc<TaskService>,
        progress: ProgressReporter,
    ) -> Result<()> {
        let active: HashSet<String> = ctx
            .sessions
            .active_session_ids()
            .into_iter()
            .collect();
        let base = ctx
            .sessions
            .base_dir();
        let mut removed = 0usize;

        for entry in super::iter_dir(base) {
            let name = entry
                .file_name()
                .to_string_lossy()
                .into_owned();
            if !active.contains(&name) {
                // Kill any orphaned ffmpeg process before removing the dir.
                #[cfg(unix)]
                if let Ok(pid_str) = std::fs::read_to_string(
                    entry
                        .path()
                        .join(".pid"),
                ) {
                    if let Ok(pid) = pid_str
                        .trim()
                        .parse::<libc::pid_t>()
                    {
                        if pid > 0 {
                            // Verify PID is actually an ffmpeg process before sending SIGKILL
                            // to guard against PID reuse across container/host restarts.
                            let is_ffmpeg = std::fs::read_to_string(format!("/proc/{pid}/comm"))
                                .map(|c| c.trim().contains("ffmpeg"))
                                .unwrap_or(false);
                            if is_ffmpeg {
                                unsafe {
                                    libc::kill(pid, libc::SIGCONT);
                                    libc::kill(pid, libc::SIGKILL);
                                }
                            }
                        }
                    }
                }
                if let Err(e) = std::fs::remove_dir_all(entry.path()) {
                    warn!(
                        "failed to remove transcode dir {}: {e:#}",
                        entry
                            .path()
                            .display()
                    );
                } else {
                    removed += 1;
                }
            }
        }
        info!(removed, "cleaned orphaned transcode dirs");

        progress.set(50.0);

        // Collect torrent IDs currently being streamed by active sessions so we
        // don't pull the rug out from under an in-progress playback.
        let mut active_torrent_ids = HashSet::new();
        for session in ctx
            .sessions
            .get_all()
        {
            if let Some(tc) = session.transcode {
                let input_url = tc
                    .read()
                    .await
                    .input_url
                    .clone();
                if let Some(id) =
                    crate::torrent::TorrentManager::torrent_id_from_url(&input_url)
                {
                    active_torrent_ids.insert(id);
                }
            }
        }

        let torrent = ctx
            .torrent
            .read()
            .await
            .clone();
        let deleted = if let Some(mgr) = torrent {
            mgr.delete_unused_with_files(&active_torrent_ids)
                .await
                .unwrap_or_else(|e| {
                    warn!("failed to clean torrents: {e:#}");
                    0
                })
        } else {
            0
        };
        info!(deleted, "cleaned torrent sessions");

        progress.set(75.0);

        // Prune extracted subtitle cache files older than 14 days so disk usage
        // stays bounded across long-running deployments.
        let sub_cache_dir = ctx
            .config
            .data_dir
            .join("subtitle-cache");
        let max_age = std::time::Duration::from_secs(14 * 86_400);
        let now = std::time::SystemTime::now();
        let mut pruned_subs = 0usize;
        if let Ok(entries) = std::fs::read_dir(&sub_cache_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Ok(meta) = entry.metadata() {
                    if meta.is_file() {
                        if let Ok(modified) = meta.modified() {
                            if now
                                .duration_since(modified)
                                .map(|age| age > max_age)
                                .unwrap_or(false)
                            {
                                if std::fs::remove_file(&path).is_ok() {
                                    pruned_subs += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        if pruned_subs > 0 {
            info!(pruned_subs, "pruned stale subtitle-cache files");
        }

        // Sweep expired ephemeral `stream` rows whose `updated_at` is older than 2 days.
        // Never touch `stream_group` rows (permanent admin-configured Stream Group definitions),
        // exclude streams referenced by active playback sessions, and clear `streams_refreshed_at`
        // on affected parents so they re-discover streams cleanly instead of serving `no-streams.mp4`.
        let active_source_ids: HashSet<String> = ctx
            .sessions
            .get_all()
            .into_iter()
            .filter_map(|s| s.media_source_id)
            .collect();
        let sweep_res: Result<u64> = async {
            let mut tx = ctx.db.begin().await?;
            let stale: Vec<(uuid::Uuid, Option<uuid::Uuid>)> = sqlx::query_as(
                "SELECT id, parent_id FROM media \
                 WHERE kind = 'stream' \
                   AND datetime(updated_at) < datetime('now', '-2 days')",
            )
            .fetch_all(&mut *tx)
            .await?;

            let mut delete_ids: Vec<uuid::Uuid> = Vec::new();
            let mut affected_parents: HashSet<uuid::Uuid> = HashSet::new();
            for (sid, pid) in stale {
                if active_source_ids.contains(&sid.to_string())
                    || active_source_ids.contains(&sid.simple().to_string())
                {
                    continue;
                }
                delete_ids.push(sid);
                if let Some(p) = pid {
                    affected_parents.insert(p);
                }
            }

            let mut total_deleted = 0u64;
            for chunk in delete_ids.chunks(500) {
                let mut qb = sqlx::QueryBuilder::new("DELETE FROM media WHERE id IN (");
                let mut sep = qb.separated(", ");
                for id in chunk {
                    sep.push_bind(*id);
                }
                qb.push(")");
                total_deleted += qb.build().execute(&mut *tx).await?.rows_affected();
            }

            let parent_vec: Vec<uuid::Uuid> = affected_parents.into_iter().collect();
            for chunk in parent_vec.chunks(500) {
                let mut qb = sqlx::QueryBuilder::new(
                    "UPDATE media SET streams_refreshed_at = NULL WHERE id IN (",
                );
                let mut sep = qb.separated(", ");
                for pid in chunk {
                    sep.push_bind(*pid);
                }
                qb.push(")");
                qb.build().execute(&mut *tx).await?;
            }

            tx.commit().await?;
            Ok(total_deleted)
        }
        .await;
        match sweep_res {
            Ok(deleted_streams) if deleted_streams > 0 => {
                info!(deleted_streams, "swept expired ephemeral stream rows");
            }
            Ok(_) => {}
            Err(e) => {
                warn!("failed to sweep expired ephemeral stream rows: {e:#}");
            }
        }

        progress.set(100.0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{Media, MediaKind};
    use uuid::Uuid;

    #[tokio::test]
    async fn sweep_preserves_stream_groups_and_fresh_streams_and_resets_stale_parents() {
        let (_server, guard) = crate::integration_test::new_test_server()
            .await
            .unwrap();
        let ctx = guard.0.clone();

        let movie_id = Uuid::new_v4();
        let group_id = Uuid::new_v4();
        let fresh_stream_id = Uuid::new_v4();
        let stale_stream_id = Uuid::new_v4();

        let movie = Media {
            id: movie_id,
            kind: MediaKind::Movie,
            title: "Test Movie".into(),
            external_ids: crate::db::ExternalIds {
                tmdb: Some(12345),
                ..Default::default()
            },
            ..Default::default()
        };
        let group = Media {
            id: group_id,
            kind: MediaKind::StreamGroup,
            title: "4K Remux Group".into(),
            ..Default::default()
        };
        let fresh_stream = Media {
            id: fresh_stream_id,
            kind: MediaKind::Stream,
            parent_id: Some(movie_id),
            title: "Fresh Stream".into(),
            ..Default::default()
        };
        let stale_stream = Media {
            id: stale_stream_id,
            kind: MediaKind::Stream,
            parent_id: Some(movie_id),
            title: "Stale Stream".into(),
            ..Default::default()
        };
        Media::insert(&ctx.db, &[movie, group, fresh_stream, stale_stream])
            .await
            .unwrap();

        // Age stream_group and both streams' created_at to 10 days ago;
        // keep fresh_stream.updated_at recent, age stale_stream.updated_at to 5 days ago,
        // and stamp movie.streams_refreshed_at.
        sqlx::query(
            "UPDATE media SET created_at = datetime('now', '-10 days'), updated_at = datetime('now', '-10 days') WHERE id = ?",
        )
        .bind(group_id)
        .execute(&ctx.db)
        .await
        .unwrap();
        sqlx::query(
            "UPDATE media SET created_at = datetime('now', '-10 days'), updated_at = datetime('now') WHERE id = ?",
        )
        .bind(fresh_stream_id)
        .execute(&ctx.db)
        .await
        .unwrap();
        sqlx::query(
            "UPDATE media SET created_at = datetime('now', '-10 days'), updated_at = datetime('now', '-5 days') WHERE id = ?",
        )
        .bind(stale_stream_id)
        .execute(&ctx.db)
        .await
        .unwrap();
        sqlx::query("UPDATE media SET streams_refreshed_at = datetime('now') WHERE id = ?")
            .bind(movie_id)
            .execute(&ctx.db)
            .await
            .unwrap();

        let tasks = Arc::new(TaskService::new(ctx.clone()).await.unwrap());
        let progress =
            ProgressReporter::new(Arc::new(std::sync::atomic::AtomicU64::new(0)));
        CleanTranscodeFolderTask
            .run(ctx.clone(), tasks, progress)
            .await
            .unwrap();

        // StreamGroup must NEVER be deleted
        assert!(Media::get_by_id(&ctx.db, &group_id).await.unwrap().is_some());
        // Fresh-updated stream (even with 10-day-old created_at) must survive
        assert!(Media::get_by_id(&ctx.db, &fresh_stream_id).await.unwrap().is_some());
        // Stale stream must be deleted
        assert!(Media::get_by_id(&ctx.db, &stale_stream_id).await.unwrap().is_none());
        // Parent's streams_refreshed_at must be cleared
        let refreshed_at: Option<String> =
            sqlx::query_scalar("SELECT streams_refreshed_at FROM media WHERE id = ?")
                .bind(movie_id)
                .fetch_one(&ctx.db)
                .await
                .unwrap();
        assert!(refreshed_at.is_none());
    }
}
