use crate::*;

pub struct Network {
    pub input: Layer<ConstantInput>,
    hidden: Vec<Layer<CachedInput<Sigmoid>>>,
    pub output: Layer<CachedInput<Sigmoid>>,
}

impl Network {
    pub fn new(sizes: Vec<usize>) -> Self {
        let len = sizes.len();
        assert!(
            len > 2,
            "must have at least 2 elements in sizes for Network::new"
        );
        let input = Layer::default_new(sizes[0]);

        let mut hidden = vec![];
        for i in 1..len - 1 {
            let mut layer = Layer::default_new(sizes[i]);
            if hidden.is_empty() {
                layer.connect(&input);
            } else {
                layer.connect(hidden.last().expect("exists"));
            }
            hidden.push(layer);
        }

        let mut output = Layer::default_new(sizes[len - 1]);
        output.connect(hidden.last().expect("exists"));

        Self {
            input,
            hidden,
            output,
        }
    }

    pub fn set_input(&self, values: Vec<f64>) {
        assert_eq!(values.len(), self.input.len());

        for (index, const_input) in self.input.iter().enumerate() {
            const_input.replace_with(values[index]);
        }
    }

    pub fn output_values(&self) -> Vec<f64> {
        self.output.iter().map(|n| n.value()).collect()
    }

    /// Mark layers as dirty - call this after you change
    /// the inputs to the network.
    pub fn dirty(&self) {
        for layer in &self.hidden {
            layer.dirty();
        }

        self.output.dirty();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works_as_expected() {
        let network = Network::new(vec![2, 3, 1]);
        assert_eq!(network.input.len(), 2);

        assert_eq!(network.hidden.len(), 1);
        assert_eq!(network.hidden[0].len(), 3);

        assert_eq!(network.output.len(), 1);

        network.set_input(vec![0.0, 1.0]);

        let output_values = network.output_values();
        assert_eq!(output_values.len(), 1);
    }
}
