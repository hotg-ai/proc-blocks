#![no_std]

extern crate alloc;
use alloc::{sync::Arc, vec, vec::Vec};
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};
use libm::fabsf;

/// A proc-block which will take [1,*,85] dimension where
/// * is no. of prediction from the models
/// 85 -> [x-cordinate, y-cordinate, width, height, confidence_value,
/// pred_classes(80 COCO classes)]  -> 5 + 80 COCO classes and will change it to
/// [*,6] here * is the number of prediction in the image who have a confidence
/// value above threshold and 6 refer to [ x-coordinate, y-coordinate, h, w,
/// confidence value, label_index]

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
        let input: Vec<f32> = input.elements().iter().map(|&x| x).collect();

        let mut vec_2d: Vec<Vec<f32>> = Vec::new();

        let mut j = 0;
        let dimensions = 85; // [cx, cy, w, h, conf, pred_cls(80)] -> 85 (5 + 80 classes)
        let confidence_index = 4;
        let label_start_index = 5;

        // loop through all the objects and select only those objects which have
        // confidence value above threshold

        for i in 0..dim[1] {
            let index = i * dimensions;

            if j != 0 {
                let rows = vec_2d.len();
                let mut x = 0;
                for i in 0..rows {
                    if fabsf(input[index] - vec_2d[i][0]) <= 0.01
                        && fabsf(input[index + 1] - vec_2d[i][1]) <= 0.01
                    {
                        x = 1;
                        continue;
                    }
                }
                if x == 1 {
                    continue;
                }
            }
            if input[index + confidence_index] <= self.threshold {
                continue;
            }

            let (ind, value) = &input
                [index + label_start_index..index + dimensions]
                .iter()
                .enumerate()
                .fold(
                    (0, 0.0),
                    |max, (ind, &val)| {
                        if val > max.1 {
                            (ind, val)
                        } else {
                            max
                        }
                    },
                );

            if value <= &self.threshold {
                continue;
            }
            vec_2d.push(vec![]);
            vec_2d[j].push(input[index]); // x-coordinate
            vec_2d[j].push(input[index + 1]); // y-coordinate
            vec_2d[j].push(input[index + 2]); // h
            vec_2d[j].push(input[index + 3]); // w
            vec_2d[j].push(*value as f32); // label confidence values
            vec_2d[j].push(*ind as f32); // label index

            j = j + 1;
        }

        let rows = vec_2d.len();
        let columns = vec_2d[0].len();

        let elements: Arc<[f32]> = vec_2d
            .into_iter()
            .flat_map(|v: Vec<f32>| v.into_iter())
            .collect();

        Tensor::new_row_major(elements, alloc::vec![rows, columns])
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
}
