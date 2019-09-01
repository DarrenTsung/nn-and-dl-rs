use crate::input::{Input, WeightedInput};
use crate::neuron::{Neuron, NeuronInputStrategy};

/// Sigmoid outputs a value between [0, 1].
pub type Sigmoid = Neuron<SigmoidStrategy>;

pub struct SigmoidStrategy;

impl NeuronInputStrategy for SigmoidStrategy {
    fn value(&self, inputs: &[WeightedInput], bias: f64) -> f64 {
        let sum_of_inputs: f64 = inputs.iter().map(|i| i.value()).sum();
        let z = sum_of_inputs + bias;
        sigmoid(z)
    }
}

/// The logistic aka sigmoid function.
#[inline]
pub fn sigmoid(f: f64) -> f64 {
    use std::f64::consts::E;
    1.0 / (1.0 + E.powf(-f))
}

impl Default for Sigmoid {
    fn default() -> Self {
        Neuron::new(SigmoidStrategy)
    }
}
