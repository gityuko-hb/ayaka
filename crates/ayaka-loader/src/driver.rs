//! Live CUDA driver memory query, isolated on a dedicated OS thread.
//!
//! `cuMemGetInfo` requires a *current* CUDA context, and cudarc's
//! `bind_to_thread()` makes the device's primary context current **for the
//! calling OS thread only** (`cuCtxSetCurrent` is thread-local state). Under a
//! multi-threaded Tokio runtime a future may be polled on worker thread `A`,
//! suspended at an `.await`, then resumed on worker thread `B` by the
//! work-stealing scheduler. If we bound the context on `A` but the next driver
//! call runs on `B`, `B` either has no current context
//! (`CUDA_ERROR_INVALID_CONTEXT`) or a stale one, producing garbage
//! free/total readings — exactly the "data disorder" failure mode.
//!
//! We therefore confine **all** driver queries to a single dedicated OS thread
//! that binds the context and serves requests over a channel. The thread never
//! moves, so the binding is stable; requests are serialized, so there is no
//! race on the driver. Callers keep the original synchronous
//! `driver_memory_info(&Device)` signature — it now simply round-trips through
//! the process-global actor, so **no existing call site changes**
//! (`profiler.rs`, `estimate`, `strategy::gpu` are all unaffected).

use std::sync::OnceLock;

use ayaka_memory::DriverMemoryInfo;
use candle_core::Device;
use crossbeam::channel::{Sender, bounded, unbounded};

use crate::error::{LoaderError, Result};

/// A single VRAM query addressed to the driver thread, carrying a one-shot
/// reply channel back to the caller.
struct MemInfoRequest {
    device: Device,
    reply: Sender<Result<DriverMemoryInfo>>,
}

/// Owns the dedicated CUDA-driver OS thread. Every `mem_info` call is
/// serialized through it, so the CUDA context is bound on exactly one thread
/// for the process lifetime and never migrates across Tokio workers.
///
/// The `crossbeam` `Sender` is `Sync`, so a single `&CudaDriverActor` (e.g. the
/// `'static` one from [`global_driver`]) can be shared by every thread without
/// locking.
pub struct CudaDriverActor {
    // `Option` so `Drop` can close the channel (drop the sender) *before*
    // joining the worker; otherwise the worker's `recv()` would block forever
    // and `join()` would deadlock.
    tx: Option<Sender<MemInfoRequest>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl CudaDriverActor {
    /// Spawn the driver thread. Invoked once, lazily, by [`global_driver`].
    fn spawn() -> Self {
        // Unbounded so a burst of concurrent planners never blocks on a full
        // queue; the queue drains promptly because each request is a
        // microsecond-scale driver call.
        let (tx, rx) = unbounded::<MemInfoRequest>();
        let handle = std::thread::Builder::new()
            .name("ayaka-cuda-driver".to_string())
            .spawn(move || {
                // This thread — and only this thread — ever binds a CUDA
                // context. It lives until the request channel is closed.
                while let Ok(req) = rx.recv() {
                    let result = query_blocking(&req.device);
                    // The caller may have dropped its receiver (timeout/cancel);
                    // ignore the resulting send error.
                    let _ = req.reply.send(result);
                }
            })
            .expect("failed to spawn ayaka-cuda-driver thread");
        Self {
            tx: Some(tx),
            handle: Some(handle),
        }
    }

    /// Query free/total VRAM for `device`. Blocks the caller until the driver
    /// thread replies. Safe to call from any thread; from `async` code prefer
    /// [`driver_memory_info_async`] (feature `async`) so the runtime worker is
    /// not blocked on the reply.
    pub fn mem_info(
        &self,
        device: &Device,
    ) -> Result<DriverMemoryInfo> {
        // A fresh single-slot reply channel per call keeps requests independent.
        let (reply_tx, reply_rx) = bounded::<Result<DriverMemoryInfo>>(1);
        self.tx
            .as_ref()
            .ok_or(LoaderError::DriverThreadDown)?
            .send(MemInfoRequest {
                device: device.clone(),
                reply: reply_tx,
            })
            .map_err(|_| LoaderError::DriverThreadDown)?;
        reply_rx
            .recv()
            .map_err(|_| LoaderError::DriverThreadDown)?
    }
}

impl Drop for CudaDriverActor {
    fn drop(&mut self) {
        // Close the request channel FIRST so the worker's `recv()` returns
        // `Err` and the service loop exits; only then can `join()` complete.
        drop(self.tx.take());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Process-wide driver actor, mirroring `ayaka_memory::global_ledger()`.
/// Spawned on first use and kept for the process lifetime.
static DRIVER_ACTOR: OnceLock<CudaDriverActor> = OnceLock::new();

/// Access the global driver actor, spawning its thread on first use.
pub fn global_driver() -> &'static CudaDriverActor {
    DRIVER_ACTOR.get_or_init(CudaDriverActor::spawn)
}

/// Query free/total VRAM for `device` (must be a CUDA device).
///
/// Routes through the global [`CudaDriverActor`] so the CUDA context is always
/// bound on the dedicated driver thread, never on a Tokio worker. The signature
/// is unchanged from the original implementation, so existing callers are
/// unaffected.
pub fn driver_memory_info(device: &Device) -> Result<DriverMemoryInfo> {
    global_driver().mem_info(device)
}

/// `async` convenience wrapper: runs the (blocking) actor round-trip on a Tokio
/// blocking-pool thread so the calling runtime worker stays free. Correctness
/// of context isolation is already guaranteed by the actor; this only avoids
/// blocking an async worker on the reply channel.
#[cfg(feature = "async")]
pub async fn driver_memory_info_async(device: Device) -> Result<DriverMemoryInfo> {
    tokio::task::spawn_blocking(move || global_driver().mem_info(&device))
        .await
        .map_err(|_| LoaderError::DriverThreadDown)?
}

/// The actual driver call. Runs **only** on the dedicated driver thread, so the
/// thread-local context binding established here is stable for the lifetime of
/// that thread. Identical cudarc calls to the original implementation.
fn query_blocking(device: &Device) -> Result<DriverMemoryInfo> {
    match device {
        Device::Cuda(dev) => {
            use candle_core::cuda::cudarc::driver::result;
            use candle_core::cuda_backend::WrapErr;

            // Bind this device's primary context to the (stable) driver thread,
            // then read device-global free/total VRAM.
            dev.cuda_stream().context().bind_to_thread().w()?;
            let (free_bytes, total_bytes) = result::mem_get_info().w()?;
            Ok(DriverMemoryInfo {
                free_bytes,
                total_bytes,
            })
        },
        _ => Err(LoaderError::InvalidConfig(
            "driver_memory_info requires a CUDA device".to_string(),
        )),
    }
}
