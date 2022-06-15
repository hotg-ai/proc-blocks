use hotg_rune_proc_blocks::{
    guest::{
        Argument, ElementTypeConstraint, Metadata, ProcBlock, RunError, Tensor,
        TensorConstraint, TensorConstraints, TensorMetadata,
    },
    ndarray::{Array1, ArrayView1, ArrayView2},
};
use smartcore::{linalg::naive::dense_matrix::*, linear::elastic_net::*};

hotg_rune_proc_blocks::export_proc_block! {
    metadata: metadata,
    proc_block: Elastic,
}

fn metadata() -> Metadata {
    Metadata::new("Elastic Net", env!("CARGO_PKG_VERSION"))
        .with_description(
            "a linear approach for modelling the relationship between a scalar response and one or more explanatory variables",
        )
        .with_repository(env!("CARGO_PKG_REPOSITORY"))
        .with_homepage(env!("CARGO_PKG_HOMEPAGE"))
        .with_tag("regression")
        .with_tag("linear modeling")
        .with_tag("analytics")
        .with_input(TensorMetadata::new("x_train"))
        .with_input(TensorMetadata::new("y_train"))
        .with_input(TensorMetadata::new("x_test"))
        .with_output(TensorMetadata::new("y_test"))
}

/// A proc block which can perform linear regression
struct Elastic;

impl ProcBlock for Elastic {
    fn tensor_constraints(&self) -> TensorConstraints {
        TensorConstraints {
            inputs: vec![
                TensorConstraint::new(
                    "x_train",
                    ElementTypeConstraint::F64,
                    vec![0, 0],
                ),
                TensorConstraint::new(
                    "y_train",
                    ElementTypeConstraint::F64,
                    vec![0],
                ),
                TensorConstraint::new(
                    "x_test",
                    ElementTypeConstraint::F64,
                    vec![0, 0],
                ),
            ],
            outputs: vec![TensorConstraint::new(
                "y_test",
                ElementTypeConstraint::F64,
                vec![0],
            )],
        }
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
        let x_train = Tensor::get_named(&inputs, "x_train")?.view_2d()?;
        let y_train = Tensor::get_named(&inputs, "y_train")?.view_1d()?;
        let x_test = Tensor::get_named(&inputs, "x_test")?.view_2d()?;

        let output = transform(x_train, y_train, x_test)?;

        Ok(vec![Tensor::new("y_test", &output)])
    }
}

impl From<Vec<Argument>> for Elastic {
    fn from(_: Vec<Argument>) -> Self { Elastic }
}

fn transform(
    x_train: ArrayView2<'_, f64>,
    y_train: ArrayView1<'_, f64>,
    x_test: ArrayView2<'_, f64>,
) -> Result<Array1<f64>, RunError> {
    // Note: we need to copy our values because elasticnet doesn't interoperate
    // with ndarray and it can't use &[T] slices.

    let (rows, columns) = x_train.dim();
    let x_train =
        DenseMatrix::new(rows, columns, x_train.into_iter().copied().collect());

    let y_train: Vec<_> = y_train.to_vec();

    let model = ElasticNet::fit(&x_train, &y_train, Default::default())
        .map_err(RunError::other)?;

    let (rows, columns) = x_test.dim();
    let x_test =
        DenseMatrix::new(rows, columns, x_test.into_iter().copied().collect());

    model
        .predict(&x_test)
        .map(Array1::from_vec)
        .map_err(RunError::other)
}

#[cfg(test)]
mod tests {
    use hotg_rune_proc_blocks::ndarray::{self, Array2};

    use super::*;

    #[test]
    fn check_model() {
        let x_train: Array2<f64> = ndarray::array![
            [234.289, 235.6, 159.0, 107.608, 1947., 60.323],
            [259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            [258.054, 368.2, 161.6, 109.773, 1949., 60.171],
            [284.599, 335.1, 165.0, 110.929, 1950., 61.187],
            [328.975, 209.9, 309.9, 112.075, 1951., 63.221],
            [346.999, 193.2, 359.4, 113.270, 1952., 63.639],
            [365.385, 187.0, 354.7, 115.094, 1953., 64.989],
            [363.112, 357.8, 335.0, 116.219, 1954., 63.761],
            [397.469, 290.4, 304.8, 117.388, 1955., 66.019],
            [419.180, 282.2, 285.7, 118.734, 1956., 67.857],
            [442.769, 293.6, 279.8, 120.445, 1957., 68.169],
            [444.546, 468.1, 263.7, 121.950, 1958., 66.513],
            [482.704, 381.3, 255.2, 123.366, 1959., 68.655],
            [502.601, 393.1, 251.4, 125.368, 1960., 69.564],
            [518.173, 480.6, 257.2, 127.852, 1961., 69.331],
            [554.894, 400.7, 282.7, 130.081, 1962., 70.551],
        ];

        let y_train: Array1<f64> = ndarray::array![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6,
            108.4, 110.8, 112.6, 114.2, 115.7, 116.9
        ];

        let y_pred =
            transform(x_train.view(), y_train.view(), x_train.view()).unwrap();

        let should_be = vec![
            112.7901174966222,
            115.23028619478328,
            104.00652847960953,
            106.91893927853232,
            101.89562519168146,
            98.62225598974453,
            100.3986322888735,
            90.34439937146931,
            99.44618079637769,
            102.87598179071631,
            103.51961064304874,
            92.90632404596613,
            101.22197835350744,
            101.6134669106201,
            95.40896231278623,
            99.70071085566008,
        ];

        assert_eq!(y_pred.to_vec(), should_be);
    }
}
