use crate::{
    proc_block_v1::{GraphError, KernelError},
    runtime_v1::{
        supported_shapes, Dimensions, ElementType, GraphContext, KernelContext,
        Metadata, TensorMetadata, TensorParam, TensorResult,
    },
};
use hotg_rune_proc_blocks::{ndarray::ArrayViewMut1, BufferExt, SliceExt};

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
        samples.set_description("All samples to perform an median on.");
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

        ctx.add_input_tensor(
            "samples",
            ElementType::F64,
            Dimensions::Fixed(&[0]),
        );
        ctx.add_output_tensor(
            "median",
            ElementType::F64,
            Dimensions::Fixed(&[1]),
        );

        Ok(())
    }

    fn kernel(id: String) -> Result<(), KernelError> {
        let ctx = KernelContext::for_node(&id).unwrap();

        let TensorResult {
            element_type,
            mut buffer,
            dimensions,
        } = ctx.get_input_tensor("samples").unwrap();

        let mut samples: ArrayViewMut1<f64> = match element_type {
            ElementType::F64 => buffer
                .view_mut(&dimensions)
                .unwrap()
                .into_dimensionality()
                .unwrap(),
            _ => panic!("Handle invalid element type"),
        };
        samples
            .as_slice_mut()
            .unwrap()
            .sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if samples.len() % 2 == 1 {
            let idx = (samples.len() / 2);
            samples[idx]

        } else {
            let idx = (samples.len() / 2);
            (samples[idx] + samples[idx+1]) / 2.0
        };
        ctx.set_output_tensor(
            "median",
            TensorParam {
                element_type: ElementType::F64,
                dimensions: &[1],
                buffer: [median].as_bytes(),
            },
        );

        Ok(())
    }
}
