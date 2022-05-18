#![cfg_attr(not(feature = "metadata"), no_std)]

#[macro_use]
extern crate alloc;

use alloc::vec::Vec;
use core::cmp::Ordering;
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};
use libm::fabsf;

/// A proc-block which takes 3-d tensor `[1, num_detection, detection_box(x, y,
/// w, h) + confidence_scores + total_detection_classes]` and filter the
/// detected objects to:
/// 1. remove duplicate detection for a single object
/// 2. remove the objects with low confidence based on a threshold
///
/// giving a 2-d tensor with dimension `[*, 6]` (where * is total number of
/// detected objects,  and 6 -> `[ x-coordinate, y-coordinate, h, w,
/// confidence_value, label_index]`) as output.
#[derive(Debug, Clone, PartialEq, ProcBlock)]
pub struct ObjectFilter {
    threshold: f32,
}

impl ObjectFilter {
    pub const fn new() -> Self { ObjectFilter { threshold: 0.7 } }
}

impl Default for ObjectFilter {
    fn default() -> Self { ObjectFilter::new() }
}

impl Transform<Tensor<f32>> for ObjectFilter {
    type Output = Tensor<f32>;

    fn transform(&mut self, input: Tensor<f32>) -> Tensor<f32> {
        let dim = input.dimensions();
        let rectangles = input.view::<3>().expect("a 3-d tensor");
        let mut objects: Vec<Object> = (0..dim[1])
            .map(|object_index| {
                rectangles.slice::<1>(&[0, object_index]).unwrap()
            })
            .filter(|view| view[4] > self.threshold)
            .map(|view| Object::from_row(view.elements()))
            .collect();

        while let Some((first, second)) = find_duplicate(&objects) {
            if objects[first].confidence > objects[second].confidence {
                objects.remove(second);
            } else {
                objects.remove(first);
            }
        }

        let rows = objects.len();
        let elements = objects
            .into_iter()
            .flat_map(|j| j.into_elements())
            .collect();

        Tensor::new_row_major(elements, vec![rows, 6])
    }
}

#[derive(Debug, Copy, Clone)]
struct Object {
    x: f32,
    y: f32,
    height: f32,
    width: f32,
    confidence: f32,
    index: usize,
}

impl Object {
    pub fn from_row(slice: &[f32]) -> Self {
        match *slice {
            [x, y, height, width, _, ref labels @ ..] => {
                let (index, confidence) = labels
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by(|a, b| {
                        a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
                    })
                    .expect("there should be at least one label");

                Object {
                    x,
                    y,
                    height,
                    width,
                    confidence,
                    index,
                }
            },
            _ => panic!("expected at least 5 elements"),
        }
    }

    fn is_duplicated(&self, other: &Object, threshold: f32) -> bool {
        fabsf(self.x - other.x) <= threshold
            && fabsf(self.y - other.y) <= threshold
    }

    fn into_elements(self) -> impl IntoIterator<Item = f32> {
        let Object {
            x,
            y,
            height,
            width,
            confidence,
            index,
        } = self;
        [x, y, height, width, confidence, index as f32]
    }
}

fn find_duplicate(objects: &[Object]) -> Option<(usize, usize)> {
    for i in 0..objects.len() {
        for j in i + 1..objects.len() {
            if objects[i].is_duplicated(&objects[j], 0.01) {
                return Some((i, j));
            }
        }
    }

    None
}

#[cfg(feature = "metadata")]
pub mod metadata {
    wit_bindgen_rust::import!("../wit-files/rune/runtime-v1.wit");
    wit_bindgen_rust::export!("../wit-files/rune/rune-v1.wit");

    struct RuneV1;

    impl rune_v1::RuneV1 for RuneV1 {
        fn start() {
            use runtime_v1::*;

            let metadata =
                Metadata::new("Object Filter", env!("CARGO_PKG_VERSION"));
            metadata.set_description(
                "Given a set of detected objects and their locations, remove duplicates and any objects below a certain threshold.",
            );
            metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
            metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
            metadata.add_tag("image");
            metadata.add_tag("classify");

            let threshold = ArgumentMetadata::new("threshold");
            threshold.set_description(
                "The minimum confidence value for an object to be included.",
            );
            threshold.set_type_hint(TypeHint::Float);
            threshold.set_default_value("0.7");
            metadata.add_argument(&threshold);

            let input = TensorMetadata::new("bounding_boxes");
            input.set_description("An arbitrary length tensor of detections, where each row starts with `[x, y, height, width, max_confidence, ...]` followed by an arbitrary number of confidence values (one value for each object type being detected).");
            let hint = supported_shapes(
                &[ElementType::Float32],
                Dimensions::Fixed(&[1, 0, 0]),
            );
            input.add_hint(&hint);
            metadata.add_input(&input);

            let output = TensorMetadata::new("normalized");
            output.set_description("The filtered objects and their indices as a list of objects, where each row contains `[x, y, height, width, confidence, index]`.");
            let hint = supported_shapes(
                &[ElementType::Float32],
                Dimensions::Fixed(&[0, 5]),
            );
            output.add_hint(&hint);
            metadata.add_output(&output);

            register_node(&metadata);
        }
    }
}

#[cfg(test)]

mod test {
    use super::*;

    #[test]
    fn test_object_filter() {
        let v: Tensor<f32> = [[[
            0.27335986, 0.43181776, 0.40072349, 0.33026114, 0.75, 0.1849257,
            0.8824799, 0.26666544, 0.10702547, 0.34699273, 0.27335986,
            0.43181776, 0.40072349, 0.33026114, 0.63204721, 0.2141086,
            0.58288711, 0.08516971, 0.33079992, 0.0627511, 0.51991946,
            0.44326415, 0.09308417, 0.25098184, 0.64352701, 0.48809405,
            0.35556684, 0.23886549, 0.15850841, 0.61959053, 0.62318601,
            0.34463603, 0.07799015, 0.33482861, 0.22496075, 0.58609099,
            0.12996288, 0.47061749, 0.56641317, 0.49165747, 0.3426614,
            0.45904443, 0.07293156, 0.2054915, 0.45656552, 0.36487279,
            0.62364449, 0.32963318, 0.35004969, 0.14574761, 0.44673359,
            0.29083161, 0.2129067, 0.41462883, 0.33459402, 0.17917575,
            0.09818682, 0.04437961, 0.5769604, 0.34821418, 0.44926693,
            0.4287493, 0.2332583, 0.29233373, 0.5974608, 0.02897593,
            0.09843597, 0.44231495, 0.30452269, 0.56230679, 0.0113074,
            0.56081945, 0.53853333, 0.43793348, 0.17007934, 0.35080665,
            0.05898283, 0.05127876, 0.29145357, 0.59377787, 0.51103643,
            0.13517603, 0.19269662, 0.47548843, 0.20795399,
        ]]]
        .into();
        let mut objects = ObjectFilter::default();
        let output = objects.transform(v);
        let should_be: Tensor<f32> = [[
            0.27335986, 0.43181776, 0.40072349, 0.33026114, 0.8824799, 1.0,
        ]]
        .into();
        assert_eq!(output, should_be);
    }

    #[test]
    fn find_the_duplicates() {
        let obj = Object {
            x: 0.5,
            y: 0.5,
            width: 1.0,
            height: 1.0,
            confidence: 1.0,
            index: 0,
        };
        let objects = vec![obj, obj];

        let duplicate_indices = find_duplicate(&objects).unwrap();

        assert_eq!(duplicate_indices, (0, 1));
    }
}
