use opencv::core::Vector;

pub fn calculate_mse(embedding1: &Vector<f32>, embedding2: &Vector<f32>) -> f32 {
    if embedding1.len() != embedding2.len() {
        panic!("Embeddings must have the same length for MSE calculation");
    }

    let mut mse = 0.0;
    for i in 0..embedding1.len() {
        let diff = embedding1.as_slice()[i] - embedding2.as_slice()[i];
        mse += diff * diff;
    }
    mse /= embedding1.len() as f32;

    mse
}

pub fn calculate_cosine_similarity(embedding1: &Vector<f32>, embedding2: &Vector<f32>) -> f32 {
    if embedding1.len() != embedding2.len() {
        panic!("Embeddings must have the same length for cosine similarity calculation");
    }

    let mut dot_product = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;

    for i in 0..embedding1.len() {
        let value1 = embedding1.as_slice()[i];
        let value2 = embedding2.as_slice()[i];

        dot_product += value1 * value2;
        norm1 += value1 * value1;
        norm2 += value2 * value2;
    }

    let magnitude1 = norm1.sqrt();
    let magnitude2 = norm2.sqrt();

    if magnitude1 == 0.0 || magnitude2 == 0.0 {
        panic!("Cannot calculate cosine similarity for zero-length vectors");
    }

    dot_product / (magnitude1 * magnitude2)
}