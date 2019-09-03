use crate::*;
use rand::Rng;
use rand_distr::StandardNormal;

pub struct Layer<N> {
    items: Vec<N>,
}

impl<N> Layer<N> {
    pub fn new(items: Vec<N>) -> Self {
        Self { items }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }
}

impl<N> std::ops::Index<usize> for Layer<N> {
    type Output = N;

    fn index(&self, i: usize) -> &Self::Output {
        self.items.index(i)
    }
}

impl<S> Layer<Neuron<S>> {
    pub fn connect(&mut self, other: &Layer<impl Input + 'static>) {
        for neuron in &mut self.items {
            for other_item in &other.items {
                let weight: f64 = rand::thread_rng().sample(StandardNormal);
                neuron.add_input(other_item, weight);
            }
        }
    }
}

impl<N: Default> Layer<N> {
    pub fn default_new(size: usize) -> Self {
        let mut items = vec![];
        for _ in 0..size {
            items.push(N::default());
        }
        Self { items }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::ConstantInput;

    #[test]
    fn it_works_as_expected() {
        let input = Layer::new(vec![
            ConstantInput::new(0.1),
            ConstantInput::new(0.3),
            ConstantInput::new(1.0),
            ConstantInput::new(0.75),
            ConstantInput::new(0.0),
        ]);

        let mut hidden: Layer<Sigmoid> = Layer::default_new(10);
        hidden.connect(&input);

        let mut output: Layer<Sigmoid> = Layer::default_new(10);
        output.connect(&hidden);

        assert!(output[0].value() >= 0.0);
    }
}
