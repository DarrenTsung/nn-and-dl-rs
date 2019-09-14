use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::NeuronLike;

pub trait Input: BoxClone {
    fn value(&self) -> f64;
}

/// This trait is required to allow cloning Input trait objects (Box<dyn Input>).
pub trait BoxClone {
    fn box_clone(&self) -> Box<dyn Input>;
}

impl<T: Input + Clone + 'static> BoxClone for T {
    fn box_clone(&self) -> Box<dyn Input> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Input> {
    fn clone(&self) -> Box<dyn Input> {
        self.box_clone()
    }
}

impl<T: Input + 'static> Input for Arc<T> {
    fn value(&self) -> f64 {
        self.as_ref().value()
    }
}

/// ConstantInput returns a value between [0, 1] as input.
#[derive(Clone)]
pub struct ConstantInput(Arc<AtomicF64Between01>);

impl ConstantInput {
    pub fn new(value: impl Into<f64>) -> Self {
        Self(Arc::new(AtomicF64Between01::new(value)))
    }

    pub fn replace_with(&self, value: impl Into<f64>) {
        self.0.store(value);
    }
}

impl Default for ConstantInput {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl Input for ConstantInput {
    fn value(&self) -> f64 {
        self.0.load()
    }
}

/// AtomicF64 implemented via AtomicU64 which is limited between
/// the range of [0, 1].
struct AtomicF64Between01(AtomicU64);

impl AtomicF64Between01 {
    /// This controls the precision since we're storing it as a u64
    const CONVERSION_FACTOR: f64 = 1_000_000_000_000.0;

    pub fn new(value: impl Into<f64>) -> Self {
        let instance = Self(AtomicU64::new(0));
        instance.store(value);
        instance
    }

    pub fn store(&self, value: impl Into<f64>) {
        let value = value.into();
        assert!(
            value >= 0.0 && value <= 1.0,
            "value must be within [0, 1], got: {}",
            value
        );
        let value = (value * Self::CONVERSION_FACTOR) as u64;
        self.0.store(value, Ordering::SeqCst);
    }

    pub fn load(&self) -> f64 {
        self.0.load(Ordering::SeqCst) as f64 / Self::CONVERSION_FACTOR
    }
}

/// WeightedInput wraps an input source with a weight.
#[derive(Clone)]
pub struct WeightedInput {
    weight: f64,
    input: Box<dyn Input>,
}

impl WeightedInput {
    pub fn new(input: &(impl Input + 'static), weight: impl Into<f64>) -> Self {
        Self {
            weight: weight.into(),
            input: input.box_clone(),
        }
    }
}

impl Input for WeightedInput {
    fn value(&self) -> f64 {
        self.input.value() * self.weight
    }
}

#[derive(Clone)]
pub struct CachedInput<I> {
    input: I,
    cached_value: Arc<AtomicF64Between01>,
    is_cached: Arc<AtomicBool>,
}

impl<I> CachedInput<I> {
    pub fn new(input: I) -> Self {
        Self {
            input,
            cached_value: Arc::new(AtomicF64Between01::new(0.0)),
            is_cached: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn dirty(&self) {
        self.is_cached.store(false, Ordering::SeqCst);
    }
}

impl<I: Default> Default for CachedInput<I> {
    fn default() -> Self {
        Self::new(I::default())
    }
}

impl<I: Input + Clone + 'static> Input for CachedInput<I> {
    fn value(&self) -> f64 {
        if self.is_cached.load(Ordering::SeqCst) {
            return self.cached_value.load();
        }

        let value = self.input.value();
        self.cached_value.store(value);
        self.is_cached.store(true, Ordering::SeqCst);
        value
    }
}

impl<I: NeuronLike> NeuronLike for CachedInput<I> {
    fn add_input(&mut self, input: &(impl Input + 'static), weight: impl Into<f64>) {
        self.input.add_input(input, weight);
    }
}
