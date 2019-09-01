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
    use std::sync::Arc;

    macro_rules! nand {
        ($($input:expr),* $(,)*) => {
            {
                let mut nand = Perceptron::default().with_bias(3);
                $(
                    nand = nand.and_input($input, -2);
                )*
                Arc::new(nand)
            }
        };
    }

    macro_rules! nand_arc {
        ($($input:expr),* $(,)*) => {
            nand!($( Arc::clone(&$input), )*)
        };
    }

    #[test]
    fn nand_perceptron_works() {
        macro_rules! binary_nand {
            ($($input:expr),* $(,)*) => {
                nand!($( BinaryInput::new($input), )*)
            };
        }

        assert_eq!(binary_nand!(0, 0).value(), 1.0);
        assert_eq!(binary_nand!(0, 1).value(), 1.0);
        assert_eq!(binary_nand!(1, 0).value(), 1.0);
        assert_eq!(binary_nand!(1, 1).value(), 0.0);
    }

    #[test]
    fn add_circuit_works() {
        struct AddCircuit {
            input_0: Arc<BinaryInput>,
            input_1: Arc<BinaryInput>,

            sum: Arc<Perceptron>,
            carry: Arc<Perceptron>,
        }

        impl AddCircuit {
            fn set_inputs(&self, input_0: usize, input_1: usize) {
                self.input_0.replace_with(input_0);
                self.input_1.replace_with(input_1);
            }
        }

        let c = {
            // i0        a0
            //      mid         sum
            // i1        a1
            //         carry
            let input_0 = Arc::new(BinaryInput::new(0));
            let input_1 = Arc::new(BinaryInput::new(0));
            let mid = nand_arc!(&input_0, &input_1);
            let a0 = nand_arc!(&input_0, &mid);
            let a1 = nand_arc!(&input_1, &mid);
            let sum = nand_arc!(&a0, &a1);
            let carry = nand_arc!(&mid, &mid);
            AddCircuit {
                input_0,
                input_1,
                sum,
                carry,
            }
        };

        c.set_inputs(0, 0);
        assert_eq!(c.sum.value(), 0.0);
        assert_eq!(c.carry.value(), 0.0);

        c.set_inputs(1, 0);
        assert_eq!(c.sum.value(), 1.0);
        assert_eq!(c.carry.value(), 0.0);

        c.set_inputs(0, 1);
        assert_eq!(c.sum.value(), 1.0);
        assert_eq!(c.carry.value(), 0.0);

        c.set_inputs(1, 1);
        assert_eq!(c.sum.value(), 0.0);
        assert_eq!(c.carry.value(), 1.0);
    }
}
