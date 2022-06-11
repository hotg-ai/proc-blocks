use crate::guest::{KernelError, Tensor, TensorConstraints};

/// The implementation of a processing block.
pub trait ProcBlock {
    fn tensor_constraints(&self) -> TensorConstraints;
    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, KernelError>;
}

impl<N: ProcBlock + ?Sized> ProcBlock for Box<N> {
    fn tensor_constraints(&self) -> TensorConstraints {
        (**self).tensor_constraints()
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, KernelError> {
        (**self).run(inputs)
    }
}
