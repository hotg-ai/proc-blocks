use anyhow::{Context, Error};
use rand::Rng;
use wasmer::{ImportObject, Module, Store, WasmerEnv};

use crate::{
    proc_block_v2::{Argument, Metadata, Node, ProcBlockV2, TensorConstraints},
    runtime_v2::{self, LogMetadata, LogValueMap},
};
use std::collections::HashMap;

pub struct ProcBlockModule(ProcBlockV2);

impl ProcBlockModule {
    pub fn load(wasm: &[u8]) -> Result<Self, Error> {
        let store = Store::default();

        let module = Module::new(&store, wasm)
            .context("Unable to compile the WebAssembly module")?;

        let mut imports = ImportObject::new();
        runtime_v2::add_to_imports(&store, &mut imports, HostFunctions);

        let (glue, _instance) =
            ProcBlockV2::instantiate(&store, &module, &mut imports)
                .context("Unable to instantiate the WebAssembly module")?;

        Ok(ProcBlockModule(glue))
    }

    pub fn metadata(&self) -> Result<Metadata, Error> {
        let meta = self.0.metadata()?;
        Ok(meta)
    }

    pub fn graph(
        &self,
        args: &HashMap<String, String>,
    ) -> Result<TensorConstraints, Error> {
        let node = self.instantiate(args)?;
        let constraints = self.0.node_tensor_constraints(&node)?;

        Ok(constraints)
    }

    pub fn instantiate(
        &self,
        args: &HashMap<String, String>,
    ) -> Result<Node, Error> {
        let args: Vec<_> = args
            .into_iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .map(|(name, value)| Argument { name, value })
            .collect();

        let instance = self.0.create_node(&args)??;

        Ok(instance)
    }
}

#[derive(Debug, Default, Clone, WasmerEnv)]
struct HostFunctions;

impl crate::runtime_v2::RuntimeV2 for HostFunctions {
    fn abort(&mut self, msg: &str) {
        #[derive(Debug, thiserror::Error)]
        #[error("Abort: {_0}")]
        struct Abort(String);

        // Safety: This will only ever be called by the WebAssembly guest
        unsafe {
            wasmer::raise_user_trap(Box::new(Abort(msg.to_string())));
        }
    }

    #[tracing::instrument(skip_all, level = "debug")]
    fn is_enabled(&mut self, _metadata: LogMetadata<'_>) -> bool { false }

    fn log(
        &mut self,
        _metadata: LogMetadata<'_>,
        _message: &str,
        _data: LogValueMap<'_>,
    ) {
    }

    fn get_random(&mut self, buffer: &mut [u8]) {
        rand::thread_rng().fill(buffer);
    }
}
