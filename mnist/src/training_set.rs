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

pub struct TrainingItemIter<S> {
    image_src: S,
    label_src: S,
    number_of_pixels: i32,
}

impl TrainingItem {
    /// Deserialize a training set for MNIST via
    /// format described: http://yann.lecun.com/exdb/mnist/
    pub fn iter_from_files(
        image_file: impl AsRef<Path>,
        label_file: impl AsRef<Path>,
    ) -> Result<TrainingItemIter<File>, failure::Error> {
        Self::iter_from_read(File::open(image_file)?, File::open(label_file)?)
    }

    pub fn iter_from_read<R: Read>(
        mut image_src: R,
        mut label_src: R,
    ) -> Result<TrainingItemIter<R>, failure::Error> {
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

        Ok(TrainingItemIter {
            image_src,
            label_src,
            number_of_pixels,
        })
    }
}

impl<S: Read> Iterator for TrainingItemIter<S> {
    type Item = TrainingItem;

    fn next(&mut self) -> Option<Self::Item> {
        let image = (0..self.number_of_pixels)
            .map(|_| self.image_src.read_u8().ok())
            .collect::<Option<Vec<_>>>()?;

        let label = self.label_src.read_u8().ok()?;

        Some(TrainingItem { image, label })
    }
}
