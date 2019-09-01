use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub trait Input {
    fn value(&self) -> f64;
}

impl<T: Input> Input for Arc<T> {
    fn value(&self) -> f64 {
        self.as_ref().value()
    }
}

/// BinaryInput returns a binary value as input.
pub struct BinaryInput(AtomicUsize);

impl BinaryInput {
    pub fn new(value: impl Into<i32>) -> Self {
        let value = value.into();
        assert!(value == 0 || value == 1);
        Self(AtomicUsize::new(value as usize))
    }

    pub fn replace_with(&self, value: impl Into<usize>) {
        self.0.store(value.into(), Ordering::SeqCst);
    }
}

impl Input for BinaryInput {
    fn value(&self) -> f64 {
        self.0.load(Ordering::SeqCst) as f64
    }
}

/// WeightedInput wraps an input source with a weight.
pub struct WeightedInput {
    weight: f64,
    input: Box<dyn Input>,
}

impl WeightedInput {
    pub fn new(input: impl Input + 'static, weight: impl Into<f64>) -> Self {
        Self {
            weight: weight.into(),
            input: Box::new(input),
        }
    }
}

impl Input for WeightedInput {
    fn value(&self) -> f64 {
        self.input.value() * self.weight
    }
}
