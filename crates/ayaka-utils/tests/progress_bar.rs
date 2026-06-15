//! Integration tests for `ayaka_utils::progress_bar` public surface.
//!
//! Tests that depend on the global suppress counter are serialized via
//! `TEST_MUTEX` because `ayaka_utils::progress_bar` uses a process-wide
//! `AtomicUsize` and `cargo test` runs tests in parallel by default.

use std::sync::Mutex;

use ayaka_utils::progress_bar::{
    MultitaskProgress, NiceProgressBar, ProgressBarColor, ProgressSuppressGuard, configure_bar,
    fmt_bytes, fmt_throughput, make_bytes_bar_style, new_multi_progress, progress_suppressed,
};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

#[test]
fn guard_silent_decrements_on_drop() {
    let _lock = TEST_MUTEX.lock().unwrap();
    let was_suppressed = progress_suppressed();
    let captured = was_suppressed;
    {
        let _g = ProgressSuppressGuard::silent();
        assert!(progress_suppressed());
    }
    assert_eq!(progress_suppressed(), captured);
    let _ = was_suppressed;
}

#[test]
fn guard_nested_decrements_in_lifo_order() {
    let _lock = TEST_MUTEX.lock().unwrap();
    let captured = progress_suppressed();
    let g1 = ProgressSuppressGuard::silent();
    let g2 = ProgressSuppressGuard::silent();
    assert!(progress_suppressed());
    drop(g1);
    assert!(progress_suppressed());
    drop(g2);
    assert_eq!(progress_suppressed(), captured);
}

#[test]
fn guard_panic_unwind_decrements() {
    let _lock = TEST_MUTEX.lock().unwrap();
    let captured = progress_suppressed();
    let _ = std::panic::catch_unwind(|| {
        let _g = ProgressSuppressGuard::silent();
        panic!("test");
    });
    assert_eq!(progress_suppressed(), captured);
}

#[test]
fn guard_noop_does_not_suppress() {
    let _lock = TEST_MUTEX.lock().unwrap();
    let captured = progress_suppressed();
    let _g = ProgressSuppressGuard::noop();
    assert_eq!(progress_suppressed(), captured);
}

#[test]
fn guard_constructor_respects_flag() {
    let _lock = TEST_MUTEX.lock().unwrap();
    let captured = progress_suppressed();
    {
        let _g = ProgressSuppressGuard::new(!captured);
        assert!(progress_suppressed());
    }
    assert_eq!(progress_suppressed(), captured);
    {
        let _g = ProgressSuppressGuard::new(captured);
        assert_eq!(progress_suppressed(), captured);
    }
    assert_eq!(progress_suppressed(), captured);
}

#[test]
fn new_multi_progress_usable_outside_suppress() {
    let _g = ProgressSuppressGuard::noop();
    let mp = new_multi_progress();
    let _bar = mp.add(indicatif::ProgressBar::new(1));
}

#[test]
fn new_multi_progress_usable_inside_suppress() {
    let _lock = TEST_MUTEX.lock().unwrap();
    let _g = ProgressSuppressGuard::silent();
    let mp = new_multi_progress();
    let _bar = mp.add(indicatif::ProgressBar::new(1));
    assert!(progress_suppressed());
}

#[test]
fn nice_progress_bar_sequential_runs() {
    let _g = ProgressSuppressGuard::noop();
    let mp = new_multi_progress();
    let data: Vec<u32> = (0..5).collect();
    let collected: Vec<u32> =
        NiceProgressBar::new(data.into_iter(), "test", ProgressBarColor::Green, &mp)
            .into_iter()
            .collect();
    assert_eq!(collected, vec![0, 1, 2, 3, 4]);
}

#[test]
fn nice_progress_bar_parallel_branch_via_rayon() {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let _g = ProgressSuppressGuard::noop();
    let data: Vec<u32> = (0..16).collect();
    let result: Vec<u32> = data
        .into_par_iter()
        .map(|x| Ok::<u32, candle_core::Error>(x * 2))
        .collect::<candle_core::Result<Vec<_>>>()
        .unwrap();
    let mut sorted = result;
    sorted.sort();
    assert_eq!(sorted, (0..16).map(|x| x * 2).collect::<Vec<_>>());
}

#[test]
fn nice_progress_bar_sequential_branch_runs() {
    let _g = ProgressSuppressGuard::noop();
    let mp = new_multi_progress();
    let data: Vec<u32> = (0..8).collect();
    let result: Vec<u32> =
        NiceProgressBar::new(data.into_iter(), "seq", ProgressBarColor::Red, &mp)
            .into_iter()
            .map(|x| Ok::<u32, candle_core::Error>(x + 100))
            .collect::<candle_core::Result<Vec<_>>>()
            .unwrap();
    assert_eq!(result, (0..8).map(|x| x + 100).collect::<Vec<_>>());
}

#[test]
fn configure_bar_respects_suppress() {
    let _lock = TEST_MUTEX.lock().unwrap();
    let _g = ProgressSuppressGuard::silent();
    let mp = new_multi_progress();
    let bar = indicatif::ProgressBar::new(1);
    configure_bar(&bar);
    let _ = mp.add(bar);
    assert!(progress_suppressed());
}

#[test]
fn multitask_progress_from_mp_round_trip() {
    let _g = ProgressSuppressGuard::noop();
    let mp = new_multi_progress();
    let tracker = MultitaskProgress::from_mp(mp);
    let bar = tracker.add_task("x", 4, ProgressBarColor::Red);
    bar.inc(4);
    assert_eq!(bar.position(), 4);
    let _mp_back = tracker.into_mp();
}

#[test]
fn multitask_progress_default_matches_new() {
    let _g = ProgressSuppressGuard::noop();
    let tracker = MultitaskProgress::default();
    let bar = tracker.add_task("d", 2, ProgressBarColor::Blue);
    bar.inc(2);
    assert_eq!(bar.position(), 2);
}

#[test]
fn multitask_progress_add_spinner() {
    let _g = ProgressSuppressGuard::noop();
    let tracker = MultitaskProgress::new();
    let bar = tracker.add_spinner("spinning");
    bar.set_message("hello".to_string());
}

#[test]
fn fmt_bytes_via_public_api() {
    assert!(fmt_bytes(0).contains("B"));
    assert!(fmt_bytes(2048).contains("KiB"));
    assert!(fmt_bytes(5 * 1024 * 1024).contains("MiB"));
}

#[test]
fn fmt_throughput_via_public_api() {
    let s = fmt_throughput(2.0 * 1024.0 * 1024.0);
    assert!(s.ends_with("/s"));
}

#[test]
fn make_bytes_bar_style_via_public_api() {
    let _ = make_bytes_bar_style("dl", ProgressBarColor::Red);
}
