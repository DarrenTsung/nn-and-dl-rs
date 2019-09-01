use crate::input::{Input, WeightedInput};

/// NeuronInputStrategy determines how to compute the
/// value returned by the neuron given the inputs and bias.
pub trait NeuronInputStrategy {
    fn value(&self, inputs: &[WeightedInput], bias: f64) -> f64;
}

/// A Neuron has N inputs and outputs a
/// value based on the strategy defined.
pub struct Neuron<S> {
    strategy: S,

    inputs: Vec<WeightedInput>,
    bias: f64,
}

impl<S> Neuron<S> {
    pub fn new(strategy: S) -> Self {
        Self {
            strategy,
            inputs: vec![],
            bias: 0.0,
        }
    }

    /// Modify the bias of the Neuron and return itself.
    pub fn with_bias(mut self, bias: impl Into<f64>) -> Self {
        self.bias = bias.into();
        self
    }

    /// Add an input to the Neuron and return itself.
    pub fn and_input(mut self, input: impl Input + 'static, weight: impl Into<f64>) -> Self {
        self.inputs.push(WeightedInput::new(input, weight));
        self
    }
}

impl<S: NeuronInputStrategy> Input for Neuron<S> {
    fn value(&self) -> f64 {
        self.strategy.value(self.inputs.as_slice(), self.bias)
    }
}
