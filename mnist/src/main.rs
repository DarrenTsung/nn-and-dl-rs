mod training_set;

use crate::training_set::TrainingItem;

fn main() -> Result<(), failure::Error> {
    let mut training_items = TrainingItem::iter_from_files(
        "./mnist/data/train-images-idx3-ubyte",
        "./mnist/data/train-labels-idx1-ubyte",
    )?;

    println!("5th Item: {:?}", training_items.nth(5).unwrap());
    println!("Finished with training items.");

    let _test_items = TrainingItem::iter_from_files(
        "./mnist/data/t10k-images-idx3-ubyte",
        "./mnist/data/t10k-labels-idx1-ubyte",
    )?;
    println!("Finished with test items.");

    Ok(())
}
