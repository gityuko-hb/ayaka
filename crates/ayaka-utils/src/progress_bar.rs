/// Reference: https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-core/src/utils/progress.rs
use indicatif::{
    MultiProgress, ProgressBar, ProgressBarIter, ProgressDrawTarget, ProgressIterator,
    ProgressStyle,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

static SUPPRESS_COUNT: AtomicUsize = AtomicUsize::new(0);

#[inline]
pub fn progress_suppressed() -> bool {
    SUPPRESS_COUNT.load(Ordering::Relaxed) > 0
}

/// RAII guard: suppress progress bars in scope.
/// Decrement occurs even when panic (UnwindSafe).
pub struct ProgressSuppressGuard {
    active: bool,
}

impl ProgressSuppressGuard {
    pub fn new(should_suppress: bool) -> Self {
        if should_suppress {
            SUPPRESS_COUNT.fetch_add(1, Ordering::SeqCst);
        }
        Self {
            active: should_suppress,
        }
    }

    /// Suppress unconditioned.
    pub fn silent() -> Self {
        Self::new(true)
    }

    /// Do not suppress (no-op guard, zero cost).
    pub fn noop() -> Self {
        Self::new(false)
    }
}

impl Drop for ProgressSuppressGuard {
    fn drop(&mut self) {
        if self.active {
            SUPPRESS_COUNT.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

impl std::panic::UnwindSafe for ProgressSuppressGuard {}
impl std::panic::RefUnwindSafe for ProgressSuppressGuard {}

/// Style Color preset for progress bars.
#[repr(u8)]
pub enum ProgressBarColor {
    Blue,
    Green,
    Red,
    Cyan,
    Yellow,
}

impl ProgressBarColor {
    fn name(&self) -> &'static str {
        match self {
            Self::Blue => "BLUE",
            Self::Green => "GREEN",
            Self::Red => "RED",
            Self::Cyan => "CYAN",
            Self::Yellow => "YELLOW",
        }
    }
}

fn make_bar_style(
    color: ProgressBarColor,
    label: &str,
) -> ProgressStyle {
    let c = color.name();
    ProgressStyle::default_bar()
        .template(&format!(
            "{label}: [{{elapsed_precise}}] \
             [{{bar:{width}.{c}/{c}}}] \
             {{pos}}/{{len}} (eta: {{eta}}) {{msg}}",
            width = crate::constants::PROGRESS_BAR_WIDTH,
        ))
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏ ")
}

fn make_spinner_style(label: &str) -> ProgressStyle {
    ProgressStyle::default_spinner()
        .template(&format!("{label}: {{spinner:.green}} {{msg}} {{elapsed}}"))
        .unwrap()
        .tick_strings(crate::constants::PROGRESS_TICK_CHARS)
}

/// Apply suppress target if we need it
pub fn configure_bar(bar: &ProgressBar) {
    if progress_suppressed() {
        bar.set_draw_target(ProgressDrawTarget::hidden());
    }
}

pub fn new_multi_progress() -> MultiProgress {
    let mp = MultiProgress::new();
    if progress_suppressed() {
        mp.set_draw_target(ProgressDrawTarget::hidden());
    }
    mp
}

/// Progress bar wrapper for iterators with a known length.
///
/// ```rust
/// use ayaka_utils::progress_bar::{new_multi_progress, NiceProgressBar, ProgressBarColor};
/// let items = vec![1, 2, 3];
/// let mp = new_multi_progress();
/// for item in NiceProgressBar::new(items.iter(), "Loading", ProgressBarColor::Blue, &mp) {
///     // process item
/// }
/// ```
pub struct NiceProgressBar<'a, T: ExactSizeIterator> {
    iter: T,
    label: &'static str,
    color: ProgressBarColor,
    mp: &'a MultiProgress,
}

impl<'a, T: ExactSizeIterator> NiceProgressBar<'a, T> {
    pub fn new(
        iter: T,
        label: &'static str,
        color: ProgressBarColor,
        mp: &'a MultiProgress,
    ) -> Self {
        Self {
            iter,
            label,
            color,
            mp,
        }
    }
}

impl<T: ExactSizeIterator> IntoIterator for NiceProgressBar<'_, T> {
    type Item = T::Item;
    type IntoIter = ProgressBarIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        let bar = ProgressBar::new(self.iter.len() as u64);
        bar.set_style(make_bar_style(self.color, self.label));
        configure_bar(&bar);
        self.mp.add(bar.clone());
        self.iter.progress_with(bar)
    }
}

/// Parallel iterator with progress bar. use rayon included
pub struct ParallelProgress<I> {
    iter: I,
    bar: ProgressBar,
}

impl<I: ParallelIterator> ParallelIterator for ParallelProgress<I>
where
    I::Item: Send,
{
    type Item = I::Item;
    fn drive_unindexed<C>(
        self,
        consumer: C,
    ) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        let bar = self.bar.clone();
        let iter = self.iter.inspect(move |_| {
            bar.inc(1);
        });
        iter.drive_unindexed(consumer)
    }
}

impl<I: IndexedParallelIterator> IndexedParallelIterator for ParallelProgress<I>
where
    I::Item: Send,
{
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn drive<C>(
        self,
        consumer: C,
    ) -> C::Result
    where
        C: rayon::iter::plumbing::Consumer<Self::Item>,
    {
        let bar = self.bar.clone();
        let iter = self.iter.inspect(move |_| {
            bar.inc(1);
        });
        iter.drive(consumer)
    }

    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(
        self,
        callback: CB,
    ) -> CB::Output {
        let bar = self.bar.clone();
        let iter = self.iter.inspect(move |_| {
            bar.inc(1);
        });
        iter.with_producer(callback)
    }
}

impl<'a, T> NiceProgressBar<'a, T>
where
    T: ExactSizeIterator + IntoParallelIterator + Send + Sync + 'a,
    <T as IntoParallelIterator>::Item: Send + 'a,
    T::Iter: IndexedParallelIterator<Item = <T as IntoParallelIterator>::Item> + Send,
    T: IntoParallelIterator<Item = <T as Iterator>::Item>,
{
    /// Run sequentially or in parallel using `is_parallel`.
    pub fn run<F, U>(
        self,
        is_parallel: bool,
        f: F,
    ) -> candle_core::Result<Vec<U>>
    where
        F: Fn(<T as IntoParallelIterator>::Item) -> candle_core::Result<U> + Sync + Send,
        U: Send,
    {
        if is_parallel {
            // Parallel path: rayon collect with error propagation
            let bar = {
                let b = ProgressBar::new(self.iter.len() as u64);
                configure_bar(&b);
                b
            };
            let par = ParallelProgress {
                iter: self.iter.into_par_iter(),
                bar,
            };
            par.map(f).collect()
        } else {
            // Sequential path
            self.into_iter().map(f).collect()
        }
    }
}

/// Progress bar dạng spinner cho long-running operation với unknown duration.
///
/// ```rust
/// use ayaka_utils::progress_bar::SpinnerBar;
/// let spinner = SpinnerBar::start("Downloading model weights");
/// // ... long operation ...
/// spinner.finish("Download complete");
/// ```
pub struct SpinnerBar {
    bar: ProgressBar,
}

impl SpinnerBar {
    pub fn start(label: &'static str) -> Self {
        let bar = ProgressBar::new_spinner();
        bar.set_style(make_spinner_style(label));
        bar.enable_steady_tick(Duration::from_millis(crate::constants::PROGRESS_TICK_MS));
        configure_bar(&bar);
        Self { bar }
    }

    /// Wrap an existing `indicatif::ProgressBar` (e.g. one obtained from a
    /// `MultiProgress`) so it can be driven through the `SpinnerBar` API.
    /// The caller is responsible for ensuring the bar uses a spinner style.
    pub fn from_indicatif(bar: ProgressBar) -> Self {
        configure_bar(&bar);
        Self { bar }
    }

    /// Update message displayed next to the spinner.
    pub fn set_message(
        &self,
        msg: impl Into<std::borrow::Cow<'static, str>>,
    ) {
        self.bar.set_message(msg);
    }

    /// Update speed (example: "1.2 GB/s").
    pub fn set_prefix(
        &self,
        prefix: impl Into<String>,
    ) {
        self.bar.set_prefix(prefix.into());
    }

    pub fn finish(
        self,
        msg: &'static str,
    ) {
        self.bar.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        self.bar.finish_with_message(msg);
    }

    /// Finish the spinner with a runtime-formatted message (`String`).
    /// Useful when the message includes values (e.g. file count) that are
    /// only known at runtime.
    pub fn finish_with_owned(
        self,
        msg: String,
    ) {
        self.bar.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        self.bar.finish_with_message(msg);
    }

    pub fn finish_and_clear(self) {
        self.bar.finish_and_clear();
    }
}

/// Track multiple tasks simultaneously with shared MultiProgress.
///
/// ```rust
/// use ayaka_utils::progress_bar::{MultitaskProgress, ProgressBarColor};
/// let tracker = MultitaskProgress::new();
/// let layer_bar = tracker.add_task("Loading layers", 32, ProgressBarColor::Blue);
/// let adapt_bar = tracker.add_task("Loading adapters", 8, ProgressBarColor::Green);
/// // ... spawn threads ...
/// tracker.wait_all();
/// ```
pub struct MultitaskProgress {
    mp: MultiProgress,
}

impl MultitaskProgress {
    pub fn new() -> Self {
        Self {
            mp: new_multi_progress(),
        }
    }

    /// Wrap an existing `MultiProgress` so it can be used through the
    /// `MultitaskProgress` API (multi-bar tracking). Useful when the caller
    /// already created a `MultiProgress` they want to share with other
    /// components.
    pub fn from_mp(mp: MultiProgress) -> Self {
        if progress_suppressed() {
            mp.set_draw_target(ProgressDrawTarget::hidden());
        }
        Self { mp }
    }

    /// Consume `self` and return the inner `MultiProgress` so the caller can
    /// keep using it after a `MultitaskProgress` is dropped.
    pub fn into_mp(self) -> MultiProgress {
        self.mp
    }

    pub fn add_task(
        &self,
        label: &str,
        total: u64,
        color: ProgressBarColor,
    ) -> ProgressBar {
        let bar = ProgressBar::new(total);
        bar.set_style(make_bar_style(color, label));
        configure_bar(&bar);
        self.mp.add(bar)
    }

    pub fn add_spinner(
        &self,
        label: &str,
    ) -> ProgressBar {
        let bar = ProgressBar::new_spinner();
        bar.set_style(make_spinner_style(label));
        configure_bar(&bar);
        self.mp.add(bar)
    }

    /// Block until all bars have finished.
    pub fn wait_all(self) {
        // MultiProgress automatically cleans up when all bars are finished.
        drop(self.mp);
    }
}

impl Default for MultitaskProgress {
    fn default() -> Self {
        Self::new()
    }
}

pub trait IterWithProgress<'a, T>: Iterator<Item = T> + 'a {
    fn with_progress(
        self,
        is_silent: bool,
    ) -> Box<dyn Iterator<Item = T> + 'a>
    where
        Self: Sized,
    {
        if is_silent || progress_suppressed() {
            Box::new(self)
        } else {
            use tqdm::Iter;
            Box::new(self.tqdm())
        }
    }
}

impl<'a, T: Iterator + 'a> IterWithProgress<'a, T::Item> for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suppress_guard_decrement_on_drop() {
        assert!(!progress_suppressed());
        {
            let _g = ProgressSuppressGuard::silent();
            assert!(progress_suppressed());
        }
        assert!(!progress_suppressed());
    }

    #[test]
    fn test_suppress_guard_nested() {
        let _g1 = ProgressSuppressGuard::silent();
        let _g2 = ProgressSuppressGuard::silent();
        assert!(progress_suppressed());
        drop(_g1);
        // Still suppressed because G2 hasn't dropped yet.
        assert!(progress_suppressed());
        drop(_g2);
        assert!(!progress_suppressed());
    }

    #[test]
    fn test_suppress_guard_unwind_safe() {
        // Ensure the counter is decremented even when catch_unwind is used.
        let _ = std::panic::catch_unwind(|| {
            let _g = ProgressSuppressGuard::silent();
            assert!(progress_suppressed());
            panic!("test panic");
        });
        // The counter must return to 0 after the unwind.
        assert!(!progress_suppressed());
    }

    #[test]
    fn test_noop_guard_zero_cost() {
        assert!(!progress_suppressed());
        let _g = ProgressSuppressGuard::noop();
        assert!(!progress_suppressed());
    }

    #[test]
    fn test_multitask_from_mp_round_trip() {
        let mp = new_multi_progress();
        let tracker = MultitaskProgress::from_mp(mp);
        let bar = tracker.add_task("from-mp-test", 10, ProgressBarColor::Green);
        bar.inc(5);
        assert_eq!(bar.position(), 5);
        // Consume and recover the inner MultiProgress.
        let _mp_back = tracker.into_mp();
    }
}
