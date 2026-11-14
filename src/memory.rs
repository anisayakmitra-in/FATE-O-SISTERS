// Copyright (C) 2026 ANISAYAK MITRA
// SPDX-License-Identifier: GPL-3.0-or-later

use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};

/// Global atomic hardware switch tracking active user interaction.
pub static USER_IS_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Bounded trajectory queue backed by a binary Write-Ahead Log (WAL).
pub struct MemorySupervisor {
    tx_queue: SyncSender<Vec<f32>>,
    rx_queue: Receiver<Vec<f32>>,
    wal_file: File,
    spilled_count: usize,
}

impl MemorySupervisor {
    /// Creates a `MemorySupervisor` writing WAL records to `"moira_wal.bin"`.
    pub fn new() -> std::io::Result<Self> {
        Self::with_wal_path(Path::new("moira_wal.bin"))
    }

    /// Creates a `MemorySupervisor` writing WAL records to a custom `wal_path`.
    pub fn with_wal_path(wal_path: &Path) -> std::io::Result<Self> {
        let (tx_queue, rx_queue) = sync_channel(64);
        let wal_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(wal_path)?;

        Ok(Self {
            tx_queue,
            rx_queue,
            wal_file,
            spilled_count: 0,
        })
    }

    /// Writes a single trajectory vector as `[len: u32 LE][f32 LE...]` to the WAL.
    fn write_wal_record(&mut self, trajectory: &[f32]) -> std::io::Result<()> {
        let len = trajectory.len() as u32;
        self.wal_file.write_all(&len.to_le_bytes())?;
        for &val in trajectory {
            self.wal_file.write_all(&val.to_le_bytes())?;
        }
        self.wal_file.flush()?;
        self.spilled_count += 1;
        Ok(())
    }

    /// Enqueues a trajectory if idle and capacity allows; otherwise spills cleanly to WAL.
    pub fn secure_trajectory(&mut self, trajectory: Vec<f32>) -> std::io::Result<()> {
        if USER_IS_ACTIVE.load(Ordering::Relaxed) {
            return self.write_wal_record(&trajectory);
        }

        if let Err(TrySendError::Full(returned)) | Err(TrySendError::Disconnected(returned)) =
            self.tx_queue.try_send(trajectory)
        {
            self.write_wal_record(&returned)?;
        }

        Ok(())
    }

    /// Drains all queued trajectories into the WAL on shutdown or checkpoint.
    pub fn flush_queue_to_wal(&mut self) -> std::io::Result<usize> {
        let mut flushed = 0usize;
        while let Ok(item) = self.rx_queue.try_recv() {
            self.write_wal_record(&item)?;
            flushed += 1;
        }
        Ok(flushed)
    }

    /// Returns total records spilled/written to the WAL.
    #[allow(dead_code)]
    pub fn spilled_count(&self) -> usize {
        self.spilled_count
    }
}

#[cfg(test)]
mod tests {
    use super::{MemorySupervisor, USER_IS_ACTIVE};
    use std::sync::atomic::Ordering;

    #[test]
    fn test_memory_supervisor_wal_spill_and_flush() {
        let tmp = std::env::temp_dir().join("moira_test_wal.bin");
        let _ = std::fs::remove_file(&tmp);

        USER_IS_ACTIVE.store(false, Ordering::Relaxed);
        let mut sup = MemorySupervisor::with_wal_path(&tmp).expect("supervisor initializes");
        sup.secure_trajectory(vec![1.0, 2.0, 3.0])
            .expect("enqueue succeeds");
        let flushed = sup.flush_queue_to_wal().expect("flush succeeds");
        assert_eq!(flushed, 1);
        assert_eq!(sup.spilled_count(), 1);

        let meta = std::fs::metadata(&tmp).expect("wal file exists");
        assert_eq!(meta.len(), 16);

        let _ = std::fs::remove_file(&tmp);
    }
}
