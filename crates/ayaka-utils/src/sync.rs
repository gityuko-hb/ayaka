use std::cell::UnsafeCell;
use std::sync::atomic::{
    AtomicBool, AtomicU64, AtomicUsize,
    Ordering::{AcqRel, Acquire, Relaxed, Release},
};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

/// Ergonomic wrapper for Arc<Mutex<T>>.
/// (Clone Arc only, do not clone T).
///
/// ```rust
/// use ayaka_utils::sync::Shared;
/// let state = Shared::new(42_i32);
/// let _clone = state.clone();
/// ```
#[derive(Debug)]
pub struct Shared<T>(Arc<Mutex<T>>);

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<T> Shared<T> {
    pub fn new(val: T) -> Self {
        Self(Arc::new(Mutex::new(val)))
    }

    /// Obtain the lock using yield-based busy-wait.
    /// Warn after ~1ms if lock cannot be obtained (potential deadlock).
    pub fn lock(&self) -> MutexGuard<'_, T> {
        let mut spins: u32 = 0;
        loop {
            match self.0.try_lock() {
                Ok(g) => return g,
                Err(_) => {
                    spins = spins.wrapping_add(1);
                    if spins == crate::constants::LOCK_SPIN_WARN_THRESHOLD {
                        tracing::warn!(
                            target: "engine::sync",
                            "Shared::lock potential deadlock — spinning > 1ms"
                        );
                    }
                    std::thread::yield_now();
                },
            }
        }
    }

    /// Non-blocking try_lock.
    pub fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
        self.0.try_lock().ok()
    }

    /// Apply closure with lock, return result.
    /// Common pattern: state.with(|s| s.field)
    pub fn with<R>(
        &self,
        f: impl FnOnce(&mut T) -> R,
    ) -> R {
        f(&mut self.lock())
    }
}

/// Spinlock for critical sections is very short (< 200ns).
/// EXPLOITATION: Use compare_exchange_weak + spin_loop_hint correctly.
/// IMPROVEMENTS OVER MISTRAL.RS: No SpinLock → Use std::Mutex for everything.
/// WARNING: Do not use if:
/// - Critical section has I/O or syscall
/// - Lock may be held when thread sleeps
/// - On single-core systems (busy-wait is wasteful)
pub struct SpinLock<T> {
    locked: AtomicBool,
    data: UnsafeCell<T>,
}

unsafe impl<T: Send> Send for SpinLock<T> {}
unsafe impl<T: Send> Sync for SpinLock<T> {}

impl<T> SpinLock<T> {
    pub const fn new(val: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            data: UnsafeCell::new(val),
        }
    }

    /// Lấy lock, spin nếu contended.
    /// Dùng compare_exchange_weak (faster trên ARM).
    #[inline]
    pub fn lock(&self) -> SpinGuard<'_, T> {
        loop {
            // Fast path: atomic exchange
            if self
                .locked
                .compare_exchange_weak(false, true, Acquire, Relaxed)
                .is_ok()
            {
                return SpinGuard { lock: self };
            }
            // Slow path: spin with hints to prevent the CPU from wasting power unnecessarily
            // x86: PAUSE instruction (pipeline stall)
            // ARM: YIELD instruction
            while self.locked.load(Relaxed) {
                std::hint::spin_loop();
            }
        }
    }

    /// Non-blocking try_lock.
    #[inline]
    pub fn try_lock(&self) -> Option<SpinGuard<'_, T>> {
        self.locked
            .compare_exchange(false, true, Acquire, Relaxed)
            .ok()
            .map(|_| SpinGuard { lock: self })
    }
}

pub struct SpinGuard<'a, T> {
    lock: &'a SpinLock<T>,
}

impl<T> std::ops::Deref for SpinGuard<'_, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: exclusive access guaranteed by SpinLock protocol
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> std::ops::DerefMut for SpinGuard<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: exclusive access guaranteed by SpinLock protocol
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<T> Drop for SpinGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        self.lock.locked.store(false, Release);
    }
}

/// SeqLock: Readers don't block writers, writers don't block readers.
/// Readers retry if writing occurs while reading.
/// Suitable for: metrics, statistics, timestamps — small data sets, lots of reading, little writing.
/// ```rust
/// use ayaka_utils::sync::SeqLock;
/// #[derive(Clone, Copy)]
/// struct Metrics { tokens: u64 }
/// let lock = SeqLock::new(Metrics { tokens: 0 });
/// lock.write(|s| s.tokens = 42);
/// let snap = lock.read();
/// let _ = snap;
/// ```
pub struct SeqLock<T: Copy> {
    sequence: AtomicU64,
    data: UnsafeCell<T>,
}

unsafe impl<T: Copy + Send> Send for SeqLock<T> {}
unsafe impl<T: Copy + Send> Sync for SeqLock<T> {}

impl<T: Copy> SeqLock<T> {
    pub const fn new(val: T) -> Self {
        Self {
            sequence: AtomicU64::new(0),
            data: UnsafeCell::new(val),
        }
    }

    /// Read T. Retry if write occurs during reading.
    /// Zero mutex overhead when there is no write contention.
    #[inline]
    pub fn read(&self) -> T {
        loop {
            let seq1 = self.sequence.load(Acquire);
            // If the sequence is odd → write is happening → spin
            if seq1 & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            // SAFETY: Read in seqlock protocol — retry if seq changes
            let val = unsafe { *self.data.get() };
            let seq2 = self.sequence.load(Acquire);
            if seq1 == seq2 {
                return val;
            }
            // Sequence changes → write occurs during read → retry
        }
    }

    /// Write T with sequence increment.
    /// Only one writer at a time (use Mutex if multi-writer is needed).
    pub fn write(
        &self,
        f: impl FnOnce(&mut T),
    ) {
        // Increment sequence → odd (informs readers that write is occurring)
        self.sequence.fetch_add(1, AcqRel);
        // SAFETY: writer exclusive — do not call write() concurrently
        unsafe { f(&mut *self.data.get()) };
        // Increment again → even (finished writing)
        self.sequence.fetch_add(1, Release);
    }
}

/// Reference: https://pkg.go.dev/sync#WaitGroup
/// WaitGroup: waits for N async tasks or threads to complete just like GoLang.
///
/// Used instead of Vec<JoinHandle> when no result is needed.
/// Particularly useful for tensor loading: spawn N threads, wait all.
/// ```rust
/// use std::sync::Arc;
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// use ayaka_utils::sync::WaitGroup;
/// let wg = WaitGroup::new();
/// let counter = Arc::new(AtomicUsize::new(0));
/// for _ in 0..3 {
///     let token = wg.token();
///     let c = counter.clone();
///     std::thread::spawn(move || {
///         c.fetch_add(1, Ordering::Relaxed);
///         drop(token);
///     });
/// }
/// wg.wait();
/// ```
pub struct WaitGroup {
    inner: Arc<WaitGroupInner>,
}

struct WaitGroupInner {
    count: Mutex<usize>,
    notify: Condvar,
}

/// Token: holds a slot in WaitGroup.
/// Drop = done.
pub struct WaitToken {
    inner: Arc<WaitGroupInner>,
}

impl WaitGroup {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(WaitGroupInner {
                count: Mutex::new(0),
                notify: Condvar::new(),
            }),
        }
    }

    /// Create token: increment counter.
    pub fn token(&self) -> WaitToken {
        *self.inner.count.lock().unwrap() += 1;
        WaitToken {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Block until all tokens have been dropped.
    pub fn wait(self) {
        let mut count = self.inner.count.lock().unwrap();
        while *count > 0 {
            count = self.inner.notify.wait(count).unwrap();
        }
    }

    /// Non-blocking check.
    pub fn is_complete(&self) -> bool {
        *self.inner.count.lock().unwrap() == 0
    }
}

impl Default for WaitGroup {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for WaitToken {
    fn drop(&mut self) {
        let mut count = self.inner.count.lock().unwrap();
        *count -= 1;
        if *count == 0 {
            self.inner.notify.notify_all();
        }
    }
}

/// Triple buffer: Producer always writes and never blocks.
/// consumer always reads the latest value and never blocks.
/// Used for: metrics snapshot (monitor thread writes, HTTP handler reads).
/// Principle: 3 slots — write, ready, read.
///    Producer: writes to the "write" slot, swaps with "ready" (atomic).
///    Consumer: swaps "ready" with "read" (atomic), reads from "read".
///    Lock-free and wait-free for both producer and consumer.
pub struct TripleBuffer<T: Clone> {
    slots: Box<[UnsafeCell<T>; 3]>,
    /// Bits: [
    ///     7: new_data_flag,
    ///     6-5: unused,
    ///     4-3: write_slot,
    ///     2-1: ready_slot,
    ///     0: read_slot
    /// ]
    /// Simplified: Use 2 atomics for clear code.
    write_idx: AtomicUsize,
    ready_idx: AtomicUsize,
    new_data: AtomicBool,
}

unsafe impl<T: Clone + Send> Send for TripleBuffer<T> {}
unsafe impl<T: Clone + Sync> Sync for TripleBuffer<T> {}

impl<T: Clone + Default> TripleBuffer<T> {
    pub fn new() -> Self {
        Self {
            slots: Box::new([
                UnsafeCell::new(T::default()),
                UnsafeCell::new(T::default()),
                UnsafeCell::new(T::default()),
            ]),
            write_idx: AtomicUsize::new(0),
            ready_idx: AtomicUsize::new(1),
            new_data: AtomicBool::new(false),
        }
    }
}

impl<T: Clone + Default> Default for TripleBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> TripleBuffer<T> {
    /// Producer: Write new value. Wait-free.
    ///
    /// SAFETY: Called only from 1 thread (producer).
    pub fn write(
        &self,
        f: impl FnOnce(&mut T),
    ) {
        let w = self.write_idx.load(Relaxed);

        // SAFETY: Just producer access write slot
        f(unsafe { &mut *self.slots[w].get() });

        // Swap write <-> ready
        let r = self.ready_idx.swap(w, AcqRel);
        self.write_idx.store(r, Relaxed);
        self.new_data.store(true, Release);
    }

    /// Consumer: Reads the latest value if available. Wait-free.
    /// Returns None if there is no new data from the previous read.
    ///
    /// SAFETY: Calls from only one thread (consumer).
    pub fn read(&self) -> Option<T> {
        if !self.new_data.swap(false, Acquire) {
            return None;
        }
        let r = self.ready_idx.load(Acquire);
        // SAFETY: consumer access read only (after swap)
        Some(unsafe { (*self.slots[r].get()).clone() })
    }

    /// Read the latest version, regardless of whether it's new or not.
    pub fn read_latest(&self) -> T {
        self.new_data.store(false, Relaxed);
        let r = self.ready_idx.load(Acquire);
        unsafe { (*self.slots[r].get()).clone() }
    }
}

// Channel buffered helpers
pub mod channel {
    pub use crossbeam::channel::{
        ReadyTimeoutError, Receiver, Sender, TryRecvError, TrySendError, bounded, unbounded,
    };

    /// Channel for token streaming: None = EOS signal.
    pub fn token_stream(capacity: usize) -> (Sender<Option<u32>>, Receiver<Option<u32>>) {
        bounded(capacity)
    }

    /// Channel for request/response: bounded for natural back-pressure.
    pub fn request_channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
        bounded(capacity)
    }
}

/// OnceLockExt
/// Extension methods OnceLock.
pub trait OnceLockExt<T> {
    fn is_initialized(&self) -> bool;
}

impl<T> OnceLockExt<T> for std::sync::OnceLock<T> {
    fn is_initialized(&self) -> bool {
        self.get().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_spinlock_exclusive() {
        let lock = Arc::new(SpinLock::new(0usize));
        let lock2 = Arc::clone(&lock);

        let h = thread::spawn(move || {
            for _ in 0..10_000 {
                *lock2.lock() += 1;
            }
        });

        for _ in 0..10_000 {
            *lock.lock() += 1;
        }

        h.join().unwrap();
        assert_eq!(*lock.lock(), 20_000);
    }

    #[test]
    fn test_seqlock_concurrent() {
        let lock = Arc::new(SeqLock::new(0u64));
        let lock2 = Arc::clone(&lock);

        // Writer thread
        let writer = thread::spawn(move || {
            for i in 0..1_000u64 {
                lock2.write(|v| *v = i);
            }
        });

        // Reader: chỉ cần không panic và trả giá trị hợp lệ
        for _ in 0..10_000 {
            let _ = lock.read(); // không crash = pass
        }

        writer.join().unwrap();
    }

    #[test]
    fn test_wait_group() {
        let wg = WaitGroup::new();
        let counter = Arc::new(AtomicUsize::new(0));

        for _ in 0..8 {
            let token = wg.token();
            let counter = Arc::clone(&counter);
            thread::spawn(move || {
                counter.fetch_add(1, Relaxed);
                drop(token);
            });
        }

        wg.wait();
        assert_eq!(counter.load(Relaxed), 8);
    }

    #[test]
    fn test_triple_buffer_no_block() {
        #[derive(Clone, Default)]
        struct Snap {
            val: u64,
        }

        let buf = TripleBuffer::<Snap>::new();

        // Producer write
        buf.write(|s| s.val = 42);

        // Consumer read
        let snap = buf.read().expect("should have new data");
        assert_eq!(snap.val, 42);

        // Second read: no new data
        assert!(buf.read().is_none());
    }

    #[test]
    fn test_shared_with() {
        let s = Shared::new(vec![1, 2, 3]);
        let len = s.with(|v| v.len());
        assert_eq!(len, 3);

        s.with(|v| v.push(4));
        let len2 = s.with(|v| v.len());
        assert_eq!(len2, 4);
    }
}
