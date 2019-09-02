use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

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
pub struct ConstantInput(Arc<AtomicU64>);

impl ConstantInput {
    /// This controls the precision since we're storing it as a u64
    const CONVERSION_FACTOR: f64 = 1_000_000_000_000.0;

    pub fn new(value: impl Into<f64>) -> Self {
        let instance = Self(Arc::new(AtomicU64::new(0)));
        instance.replace_with(value);
        instance
    }

    pub fn replace_with(&self, value: impl Into<f64>) {
        let value = value.into();
        assert!(
            value >= 0.0 && value <= 1.0,
            "value must be within [0, 1], got: {}",
            value
        );
        let value = (value * Self::CONVERSION_FACTOR) as u64;
        self.0.store(value, Ordering::SeqCst);
    }
}

impl Default for ConstantInput {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl Input for ConstantInput {
    fn value(&self) -> f64 {
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
