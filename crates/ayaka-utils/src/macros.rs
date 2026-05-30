/// Use MutexGuard with yield-based busy-wait.
///
/// Yields after each failure to avoid starvation.
/// After 1,000 attempts (~1ms on most hardware) → warn about possible deadlock.
///
/// ```rust
/// use ayaka_utils::lock_or_yield;
/// let m = std::sync::Mutex::new(42i32);
/// let guard = lock_or_yield!(m);
/// assert_eq!(*guard, 42);
/// ```
#[macro_export]
macro_rules! lock_or_yield {
    ($mutex:expr) => {{
        let mut _spins: u32 = 0;
        loop {
            match $mutex.try_lock() {
                Ok(guard) => break guard,
                Err(_) => {
                    _spins = _spins.wrapping_add(1);
                    if _spins == $crate::constants::LOCK_SPIN_WARN_THRESHOLD {
                        tracing::warn!(
                            target: "engine::sync",
                            location = concat!(file!(), ":", line!()),
                            "lock_or_yield: potential deadlock — held > ~1ms"
                        );
                    }
                    std::thread::yield_now();
                }
            }
        }
    }};
}

/// Similar to `lock_or_yield!` but panics if the lock isn't secured after `timeout_ms`.
///
/// Used for critical sections that shouldn't be blocked for too long.
///
/// ```rust
/// use ayaka_utils::lock_timeout;
/// let m = std::sync::Mutex::new(42i32);
/// let guard = lock_timeout!(m, 100);
/// assert_eq!(*guard, 42);
/// ```
#[macro_export]
macro_rules! lock_timeout {
    ($mutex:expr, $timeout_ms:expr) => {{
        let _deadline = std::time::Instant::now() + std::time::Duration::from_millis($timeout_ms);
        loop {
            match $mutex.try_lock() {
                Ok(guard) => break guard,
                Err(_) => {
                    if std::time::Instant::now() >= _deadline {
                        panic!(
                            "lock_timeout!: failed to acquire lock within {}ms at {}:{}",
                            $timeout_ms,
                            file!(),
                            line!()
                        );
                    }
                    std::hint::spin_loop();
                },
            }
        }
    }};
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_lock_or_yield_basic() {
        let m = Mutex::new(42);
        let guard = crate::lock_or_yield!(m);
        assert_eq!(*guard, 42);
    }

    #[test]
    fn test_lock_or_yield_contention() {
        let m = Arc::new(Mutex::new(0));
        let m2 = m.clone();
        let handle = thread::spawn(move || {
            let _guard = m2.lock().unwrap();
            thread::sleep(Duration::from_millis(50));
        });
        thread::sleep(Duration::from_millis(10));
        let guard = crate::lock_or_yield!(m);
        assert_eq!(*guard, 0);
        handle.join().unwrap();
    }

    #[test]
    fn test_lock_timeout_basic() {
        let m = Mutex::new(42);
        let guard = crate::lock_timeout!(m, 100);
        assert_eq!(*guard, 42);
    }

    #[test]
    fn test_lock_timeout_panic() {
        let m = Arc::new(Mutex::new(0));
        let m2 = m.clone();
        let handle = thread::spawn(move || {
            let _hold = m2.lock().unwrap();
            thread::sleep(Duration::from_millis(500));
        });
        thread::sleep(Duration::from_millis(50));
        let result = std::panic::catch_unwind(|| {
            let _guard = crate::lock_timeout!(m, 100);
        });
        assert!(result.is_err());
        handle.join().unwrap();
    }
}
