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
                            unsafe {
                                libc::kill(pid, libc::SIGCONT);
                                libc::kill(pid, libc::SIGKILL);
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

        // Sweep expired ephemeral stream / stream_group rows older than 2 days
        // so the media table doesn't grow unbounded with stale debrid stream rows.
        match sqlx::query(
            "DELETE FROM media \
             WHERE kind IN ('stream', 'stream_group') \
               AND datetime(created_at) < datetime('now', '-2 days')",
        )
        .execute(&ctx.db)
        .await
        {
            Ok(res) if res.rows_affected() > 0 => {
                info!(
                    deleted_streams = res.rows_affected(),
                    "swept expired ephemeral stream rows"
                );
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
