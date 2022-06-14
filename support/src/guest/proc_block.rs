use crate::guest::{RunError, Tensor, TensorConstraints};

/// The implementation of a processing block.
pub trait ProcBlock {
    fn tensor_constraints(&self) -> TensorConstraints;
    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError>;
}

impl<N: ProcBlock + ?Sized> ProcBlock for Box<N> {
    fn tensor_constraints(&self) -> TensorConstraints {
        (**self).tensor_constraints()
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
        (**self).run(inputs)
    }
}
