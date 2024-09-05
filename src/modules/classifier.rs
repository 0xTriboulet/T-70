use opencv::{core::{FileStorage, FileStorage_Mode}, objdetect::CascadeClassifier, Result};
use opencv::prelude::CascadeClassifierTrait;
use opencv::prelude::FileStorageTraitConst;

const CASCADE_XML: &str = include_str!("../../models/haarcascade_frontalface_default.xml");

pub fn init_face_classifier() -> Result<CascadeClassifier> {
    let storage = FileStorage::new_def(CASCADE_XML, i32::from(FileStorage_Mode::READ) | i32::from(FileStorage_Mode::MEMORY))?;
    let mut face_classifier = CascadeClassifier::default()?;
    face_classifier.read(&storage.get_first_top_level_node()?)?;
    Ok(face_classifier)
}
