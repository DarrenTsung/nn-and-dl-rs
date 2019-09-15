pub struct Expected<'a>(&'a [f64]);
pub struct Output<'a>(&'a [f64]);

pub trait CostFunction {
    fn cost(&self, output: Output, expected: Expected) -> f64;
}

pub struct Pow2CostFunction;

impl CostFunction for Pow2CostFunction {
    fn cost(&self, output: Output, expected: Expected) -> f64 {
        let output = output.0;
        let expected = expected.0;

        assert_eq!(output.len(), expected.len());

        let mut cost = 0.0;
        for (index, val) in output.iter().enumerate() {
            cost += (val - expected[index]).powi(2);
        }
        cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn cost_against_expected_works() {
        let output = vec![0.0, 0.5, 1.0];
        let expected = vec![1.0, 1.0, 1.0];

        let cost = Pow2CostFunction.cost(Output(&output), Expected(&expected));
        assert_abs_diff_eq!(cost, 1.25);
    }
}
