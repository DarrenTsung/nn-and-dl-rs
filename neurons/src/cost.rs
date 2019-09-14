pub trait CostLike {
    fn cost_against_expected(&self, expected: &[f64]) -> f64;
}

impl CostLike for Vec<f64> {
    fn cost_against_expected(&self, expected: &[f64]) -> f64 {
        assert_eq!(self.len(), expected.len());

        let mut cost = 0.0;
        for (index, val) in self.iter().enumerate() {
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

        let cost = output.cost_against_expected(&expected);
        assert_abs_diff_eq!(cost, 1.25);
    }
}
