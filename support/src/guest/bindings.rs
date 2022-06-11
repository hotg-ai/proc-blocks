pub use self::{proc_block_v2::*, runtime_v2::*};

use crate::guest::{logging, ProcBlock};
use wit_bindgen_rust::Handle;

wit_bindgen_rust::import!("../wit-files/rune/runtime-v2.wit");
wit_bindgen_rust::export!("../wit-files/rune/proc-block-v2.wit");

extern "Rust" {
    fn __proc_block_metadata() -> Metadata;
    fn __proc_block_new(
        args: Vec<Argument>,
    ) -> Result<Box<dyn ProcBlock>, CreateError>;
}

struct ProcBlockV2;

impl proc_block_v2::ProcBlockV2 for ProcBlockV2 {
    fn metadata() -> Metadata {
        logging::initialize_logger();
        unsafe { __proc_block_metadata() }
    }

    fn create_node(
        args: Vec<Argument>,
    ) -> Result<wit_bindgen_rust::Handle<self::Node>, CreateError> {
        logging::initialize_logger();
        let proc_block = unsafe { __proc_block_new(args)? };
        Ok(Handle::new(Node(Box::new(proc_block))))
    }
}

pub struct Node(Box<dyn ProcBlock>);

impl proc_block_v2::Node for Node {
    fn tensor_constraints(&self) -> TensorConstraints {
        self.0.tensor_constraints()
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, KernelError> {
        self.0.run(inputs)
    }
}
