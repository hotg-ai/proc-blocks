/// Tell the runtime that a WebAssembly module contains a proc-block.
#[macro_export]
macro_rules! export_proc_block {
    (metadata: $metadata_func:expr, proc_block: $proc_block:ty $(,)?) => {
        #[doc(hidden)]
        #[no_mangle]
        pub fn __proc_block_metadata() -> $crate::guest::Metadata { $metadata_func() }

        #[doc(hidden)]
        #[no_mangle]
        pub fn __proc_block_new(
            args: Vec<$crate::guest::Argument>,
        ) -> Result<Box<dyn $crate::guest::ProcBlock>, $crate::guest::CreateError> {
            fn assert_impl_proc_block(_: &impl $crate::guest::ProcBlock) {}

            let proc_block = <$proc_block>::try_from(args)?;
            assert_impl_proc_block(&proc_block);

            Ok(Box::new(proc_block) as Box<dyn $crate::guest::ProcBlock>)
        }
    };
}
