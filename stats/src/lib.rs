use crate::{
    proc_block_v1::{GraphError, KernelError},
    runtime_v1::{
        Dimensions, ElementType, GraphContext, KernelContext, Metadata, TensorMetadata,
        TensorParam, TensorResult,
    },
};
use hotg_rune_proc_blocks::{ndarray::ArrayView1, BufferExt, SliceExt};

wit_bindgen_rust::import!("../wit-files/rune/runtime-v1.wit");
wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let meta = Metadata::new("Statistics", env!("CARGO_PKG_VERSION"));

        let samples = TensorMetadata::new("samples");
        samples.set_description("All samples to perform an average on");
        meta.add_input(&samples);

        let mean = TensorMetadata::new("mean");
        mean.set_description("The mean");
        meta.add_output(&mean);

        let std_dev = TensorMetadata::new("std_dev");
        std_dev.set_description("The standard deviation.");
        meta.add_output(&std_dev);

        runtime_v1::register_node(&meta);
    }

    fn graph(id: String) -> Result<(), GraphError> {
        let ctx = GraphContext::for_node(&id).unwrap();

        ctx.add_input_tensor("samples", ElementType::F64, Dimensions::Fixed(&[0]));
        ctx.add_output_tensor("mean", ElementType::F64, Dimensions::Fixed(&[1]));
        ctx.add_output_tensor("std_dev", ElementType::F64, Dimensions::Fixed(&[1]));

        Ok(())
    }

    fn kernel(id: String) -> Result<(), KernelError> {
        let ctx = KernelContext::for_node(&id).unwrap();

        let TensorResult {
            element_type,
            buffer,
            dimensions,
        } = ctx.get_input_tensor("samples").unwrap();

        let samples: ArrayView1<f64> = match element_type {
            ElementType::F64 => buffer
                .view(&dimensions)
                .unwrap()
                .into_dimensionality()
                .unwrap(),
            _ => panic!("Handle invalid element type"),
        };
        let mean = samples.mean().unwrap();
        let std_dev = samples.std(1.0);

        ctx.set_output_tensor(
            "mean",
            TensorParam {
                element_type: ElementType::F64,
                dimensions: &[1],
                buffer: [mean].as_bytes(),
            },
        );
        ctx.set_output_tensor(
            "std_dev",
            TensorParam {
                element_type: ElementType::F64,
                dimensions: &[1],
                buffer: [std_dev].as_bytes(),
            },
        );

        Ok(())
    }
}
