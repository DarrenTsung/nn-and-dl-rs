mod training_set;

use crate::training_set::*;

fn main() -> Result<(), failure::Error> {
    let mut training_set = TrainingSet::from_files(
        "./mnist/data/train-images-idx3-ubyte",
        "./mnist/data/train-labels-idx1-ubyte",
    )?;
    for batch in training_set.batch_iter(10_000) {
        println!("Going over batch of size: {}", batch.len());
    }
    println!("Finished with training set.");

    let _test_set = TrainingSet::from_files(
        "./mnist/data/t10k-images-idx3-ubyte",
        "./mnist/data/t10k-labels-idx1-ubyte",
    )?;
    println!("Finished with test set.");

    Ok(())
}
