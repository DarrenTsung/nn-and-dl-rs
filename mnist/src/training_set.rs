use byteorder::{BigEndian, ReadBytesExt};
use failure::format_err;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug)]
pub struct TrainingItem {
    /// Pixel values are from 0-255 (0 - white, 255 - black)
    image: Vec<u8>,
    /// Value is from 0-9 (the number the image corresponds to)
    label: u8,
}

pub struct TrainingSet {
    items: Vec<TrainingItem>,
}

impl TrainingSet {
    /// Deserialize a training set for MNIST via
    /// format described: http://yann.lecun.com/exdb/mnist/
    pub fn from_files(
        image_file: impl AsRef<Path>,
        label_file: impl AsRef<Path>,
    ) -> Result<Self, failure::Error> {
        Self::from_read(File::open(image_file)?, File::open(label_file)?)
    }

    pub fn from_read(
        mut image_src: impl Read,
        mut label_src: impl Read,
    ) -> Result<Self, failure::Error> {
        if image_src.read_i32::<BigEndian>()? != 2051 {
            return Err(format_err!("Image source magic number does not match"));
        }

        if label_src.read_i32::<BigEndian>()? != 2049 {
            return Err(format_err!("Label source magic number does not match"));
        }

        let number_of_items = image_src.read_i32::<BigEndian>()?;
        if label_src.read_i32::<BigEndian>()? != number_of_items {
            return Err(format_err!(
                "Number of items doesn't match up to {}",
                number_of_items
            ));
        }

        let number_of_pixels = {
            let height = image_src.read_i32::<BigEndian>()?;
            let width = image_src.read_i32::<BigEndian>()?;
            height * width
        };

        let mut items = Vec::with_capacity(number_of_items as usize);
        for _ in 0..number_of_items {
            let mut image = vec![0u8; number_of_pixels as usize];
            image_src.read_exact(&mut image)?;

            let label = label_src.read_u8()?;

            items.push(TrainingItem { image, label });
        }

        Ok(TrainingSet { items })
    }
}

impl std::ops::Index<usize> for TrainingSet {
    type Output = TrainingItem;

    fn index(&self, i: usize) -> &Self::Output {
        &self.items[i]
    }
}
