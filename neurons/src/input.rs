pub trait Input {
    fn value(&self) -> f64;
}

/// ConstantInput returns a constant as input.
pub struct ConstantInput(f64);

impl ConstantInput {
    pub fn new(value: impl Into<f64>) -> Self {
        Self(value.into())
    }

    pub fn replace_with(&mut self, value: impl Into<f64>) {
        self.0 = value.into();
    }
}

impl Input for ConstantInput {
    fn value(&self) -> f64 {
        self.0
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
