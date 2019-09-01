use crate::input::{Input, WeightedInput};

/// Perceptron takes in N inputs and outputs a
/// value of either (0, 1) based on the configured
/// weights and bias.
pub struct Perceptron {
    inputs: Vec<WeightedInput>,
    bias: f64,
}

impl Perceptron {
    /// Modify the bias of the Perceptron and return itself.
    pub fn with_bias(mut self, bias: impl Into<f64>) -> Self {
        self.bias = bias.into();
        self
    }

    /// Add an input to the Perceptron and return itself.
    pub fn and_input(mut self, input: impl Input + 'static, weight: impl Into<f64>) -> Self {
        self.inputs.push(WeightedInput::new(input, weight));
        self
    }
}

impl Default for Perceptron {
    fn default() -> Self {
        Self {
            inputs: vec![],
            bias: 0.0,
        }
    }
}

impl Input for Perceptron {
    fn value(&self) -> f64 {
        let sum_of_inputs: f64 = self.inputs.iter().map(|i| i.value()).sum();
        if sum_of_inputs + self.bias <= 0.0 {
            0.0
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{BinaryInput, Input};

    macro_rules! nand {
        ($($input:expr),* $(,)*) => {
            {
                let mut nand = Perceptron::default().with_bias(3);
                $(
                    nand = nand.and_input($input, -2);
                )*
                nand
            }
        };
    }

    #[test]
    fn nand_perceptron_works() {
        macro_rules! binary_nand {
            ($($input:expr),* $(,)*) => {
                nand!(
                    $(
                        BinaryInput::new($input),
                    )*
                )
            };
        }

        assert_eq!(binary_nand!(0, 0).value(), 1.0);
        assert_eq!(binary_nand!(0, 1).value(), 1.0);
        assert_eq!(binary_nand!(1, 0).value(), 1.0);
        assert_eq!(binary_nand!(1, 1).value(), 0.0);
    }
}
