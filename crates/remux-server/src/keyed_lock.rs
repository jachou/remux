use dashmap::DashMap;
use std::{
    hash::Hash,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, OnceLock,
    },
};
use tokio::sync::{Mutex, OwnedMutexGuard};

pub(crate) struct KeyEntry {
    mutex: Arc<Mutex<()>>,
    pub(crate) waiters: AtomicUsize,
}

/// Per-key async mutex. Only one task at a time may hold the lock for a given key.
/// The map entry is removed automatically when the guard is dropped.
pub(crate) struct KeyedLock<K: Eq + Hash + Clone + Send + Sync + 'static> {
    map: OnceLock<Arc<DashMap<K, Arc<KeyEntry>>>>,
}

impl<K: Eq + Hash + Clone + Send + Sync + 'static> KeyedLock<K> {
    pub const fn new() -> Self {
        Self {
            map: OnceLock::new(),
        }
    }

    pub(crate) fn inner(&self) -> Arc<DashMap<K, Arc<KeyEntry>>> {
        self.map
            .get_or_init(|| Arc::new(DashMap::new()))
            .clone()
    }

    /// Acquire the lock for `key`, inserting an entry if none exists.
    pub async fn lock(&self, key: K) -> KeyedLockGuard<K> {
        let map = self.inner();
        let entry = {
            let entry_ref = map.entry(key.clone()).or_insert_with(|| {
                Arc::new(KeyEntry {
                    mutex: Arc::new(Mutex::new(())),
                    waiters: AtomicUsize::new(0),
                })
            });
            // Increment waiters WHILE holding the DashMap shard lock so a concurrent
            // Drop cannot observe waiters == 0 and evict this entry out from under us.
            entry_ref.waiters.fetch_add(1, Ordering::SeqCst);
            entry_ref.clone()
        };

        let mut waiter_guard = WaiterGuard {
            map: map.clone(),
            key: key.clone(),
            entry: entry.clone(),
            acquired: false,
        };

        let _guard = entry.mutex.clone().lock_owned().await;
        waiter_guard.acquired = true;
        drop(waiter_guard);

        KeyedLockGuard {
            map,
            key,
            entry,
            _guard: Some(_guard),
        }
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.map
            .get()
            .map_or(false, |m| m.contains_key(key))
    }

    /// Acquire the lock only if an entry already exists (someone else is working).
    /// Returns `None` immediately if no entry is found.
    pub async fn lock_if_exists(&self, key: &K) -> Option<KeyedLockGuard<K>> {
        let map = self.inner();
        let entry = {
            let entry_ref = map.get(key)?;
            entry_ref.waiters.fetch_add(1, Ordering::SeqCst);
            Arc::clone(&entry_ref)
        };

        let mut waiter_guard = WaiterGuard {
            map: map.clone(),
            key: key.clone(),
            entry: entry.clone(),
            acquired: false,
        };

        let _guard = entry.mutex.clone().lock_owned().await;
        waiter_guard.acquired = true;
        drop(waiter_guard);

        Some(KeyedLockGuard {
            map,
            key: key.clone(),
            entry,
            _guard: Some(_guard),
        })
    }
}

struct WaiterGuard<K: Eq + Hash + Clone + Send + Sync + 'static> {
    map: Arc<DashMap<K, Arc<KeyEntry>>>,
    key: K,
    entry: Arc<KeyEntry>,
    acquired: bool,
}

impl<K: Eq + Hash + Clone + Send + Sync + 'static> Drop for WaiterGuard<K> {
    fn drop(&mut self) {
        if !self.acquired {
            let prev = self.entry.waiters.fetch_sub(1, Ordering::SeqCst);
            if prev == 1 {
                self.map.remove_if(&self.key, |_, current| {
                    Arc::ptr_eq(current, &self.entry)
                        && current.waiters.load(Ordering::SeqCst) == 0
                });
            }
        }
    }
}

pub(crate) struct KeyedLockGuard<K: Eq + Hash + Clone + Send + Sync + 'static> {
    map: Arc<DashMap<K, Arc<KeyEntry>>>,
    key: K,
    entry: Arc<KeyEntry>,
    _guard: Option<OwnedMutexGuard<()>>,
}

impl<K: Eq + Hash + Clone + Send + Sync + 'static> Drop for KeyedLockGuard<K> {
    fn drop(&mut self) {
        drop(self._guard.take());
        let prev = self.entry.waiters.fetch_sub(1, Ordering::SeqCst);
        if prev == 1 {
            self.map.remove_if(&self.key, |_, current| {
                Arc::ptr_eq(current, &self.entry)
                    && current.waiters.load(Ordering::SeqCst) == 0
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::KeyedLock;
    use std::sync::{atomic::Ordering, Arc};
    use tokio::{sync::oneshot, time::Duration};

    #[tokio::test]
    async fn queued_waiter_keeps_the_key_locked_during_handoff() {
        let locks = Arc::new(KeyedLock::new());
        let first = locks
            .lock("session")
            .await;
        let (acquired_tx, acquired_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let waiting_locks = locks.clone();
        let waiter = tokio::spawn(async move {
            let _second = waiting_locks
                .lock("session")
                .await;
            acquired_tx
                .send(())
                .unwrap();
            release_rx
                .await
                .unwrap();
        });

        while locks
            .inner()
            .get("session")
            .unwrap()
            .value()
            .waiters
            .load(Ordering::SeqCst)
            < 2
        {
            tokio::task::yield_now().await;
        }

        drop(first);
        tokio::time::timeout(Duration::from_secs(1), acquired_rx)
            .await
            .unwrap()
            .unwrap();
        assert!(locks.contains_key(&"session"));
        assert!(
            tokio::time::timeout(Duration::from_millis(50), locks.lock("session"))
                .await
                .is_err()
        );

        release_tx
            .send(())
            .unwrap();
        waiter
            .await
            .unwrap();
        assert!(!locks.contains_key(&"session"));
    }

    #[tokio::test]
    async fn cancelled_waiter_does_not_leak_key_in_map() {
        let locks = Arc::new(KeyedLock::new());
        let first = locks
            .lock("session_cancel")
            .await;
        let waiting_locks = locks.clone();
        let waiter = tokio::spawn(async move {
            let _second = waiting_locks
                .lock("session_cancel")
                .await;
        });

        while locks
            .inner()
            .get("session_cancel")
            .unwrap()
            .value()
            .waiters
            .load(Ordering::SeqCst)
            < 2
        {
            tokio::task::yield_now().await;
        }

        drop(first);
        waiter.abort();
        let _ = waiter.await;
        assert!(!locks.contains_key(&"session_cancel"));
    }
}
