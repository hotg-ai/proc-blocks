use crate::{
    proc_block_v1::{GraphError, KernelError},
    runtime_v1::{
        Dimensions, ElementType, GraphContext, KernelContext, Metadata, TensorMetadata,
        TensorParam, TensorResult, supported_shapes
    },
};
use hotg_rune_proc_blocks::{ndarray::ArrayView1, BufferExt, SliceExt};

wit_bindgen_rust::import!("../wit-files/rune/runtime-v1.wit");
wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata = Metadata::new("Median", env!("CARGO_PKG_VERSION"));
        metadata.set_description(env!("CARGO_PKG_DESCRIPTION"));
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("numeric");
        metadata.add_tag("stats");
        let samples = TensorMetadata::new("samples");
        samples.set_description("All samples to perform an average on");
        let hint = supported_shapes(&[ElementType::F64], Dimensions::Dynamic);
        samples.add_hint(&hint);
        metadata.add_input(&samples);
        

        let median = TensorMetadata::new("median");
        median.set_description("The median");
        let hint = supported_shapes(&[ElementType::F64], Dimensions::Dynamic);
        median.add_hint(&hint);
        metadata.add_output(&median);
        

        runtime_v1::register_node(&metadata);
    }

    fn graph(id: String) -> Result<(), GraphError> {
        let ctx = GraphContext::for_node(&id).unwrap();

        ctx.add_input_tensor("samples", ElementType::F64, Dimensions::Fixed(&[0]));
        ctx.add_output_tensor("median", ElementType::F64, Dimensions::Fixed(&[1]));

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
        let mut median_slice: Vec<f64> = samples.to_slice().unwrap().to_vec();
            median_slice.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = (median_slice.len() as f32 / 2.0) as i32 as usize; 

        ctx.set_output_tensor(
            "median",
            TensorParam {
                element_type: ElementType::F64,
                dimensions: &[1],
                buffer: [samples[median]].as_bytes(),
            },
        );

        Ok(())
    }
}
