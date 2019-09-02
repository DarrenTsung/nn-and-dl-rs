mod training_set;

use crate::training_set::*;

fn main() -> Result<(), failure::Error> {
    let training_set = TrainingSet::from_files(
        "./mnist/data/train-images-idx3-ubyte",
        "./mnist/data/train-labels-idx1-ubyte",
    )?;
    println!("5th Item: {:?}", training_set[5]);
    println!("Finished with training set.");

    let _test_set = TrainingSet::from_files(
        "./mnist/data/t10k-images-idx3-ubyte",
        "./mnist/data/t10k-labels-idx1-ubyte",
    )?;
    println!("Finished with test set.");

    Ok(())
}
