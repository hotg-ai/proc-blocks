pub mod rune_v1 {
    #[allow(unused_imports)]
    use wit_bindgen_wasmtime::{anyhow, wasmtime};

    /// Auxiliary data associated with the wasm exports.
    ///
    /// This is required to be stored within the data of a
    /// `Store<T>` itself so lifting/lowering state can be managed
    /// when translating between the host and wasm.
    #[derive(Default)]
    pub struct RuneV1Data {}
    pub struct RuneV1<T> {
        get_state: Box<dyn Fn(&mut T) -> &mut RuneV1Data + Send + Sync>,
        start: wasmtime::TypedFunc<(), ()>,
    }
    impl<T> RuneV1<T> {
        #[allow(unused_variables)]

        /// Adds any intrinsics, if necessary for this exported wasm
        /// functionality to the `linker` provided.
        ///
        /// The `get_state` closure is required to access the
        /// auxiliary data necessary for these wasm exports from
        /// the general store's state.
        pub fn add_to_linker(
            linker: &mut wasmtime::Linker<T>,
            get_state: impl Fn(&mut T) -> &mut RuneV1Data
                + Send
                + Sync
                + Copy
                + 'static,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        /// Instantiates the provided `module` using the specified
        /// parameters, wrapping up the result in a structure that
        /// translates between wasm and the host.
        ///
        /// The `linker` provided will have intrinsics added to it
        /// automatically, so it's not necessary to call
        /// `add_to_linker` beforehand. This function will
        /// instantiate the `module` otherwise using `linker`, and
        /// both an instance of this structure and the underlying
        /// `wasmtime::Instance` will be returned.
        ///
        /// The `get_state` parameter is used to access the
        /// auxiliary state necessary for these wasm exports from
        /// the general store state `T`.
        pub fn instantiate(
            mut store: impl wasmtime::AsContextMut<Data = T>,
            module: &wasmtime::Module,
            linker: &mut wasmtime::Linker<T>,
            get_state: impl Fn(&mut T) -> &mut RuneV1Data
                + Send
                + Sync
                + Copy
                + 'static,
        ) -> anyhow::Result<(Self, wasmtime::Instance)> {
            Self::add_to_linker(linker, get_state)?;
            let instance = linker.instantiate(&mut store, module)?;
            Ok((Self::new(store, &instance, get_state)?, instance))
        }

        /// Low-level creation wrapper for wrapping up the exports
        /// of the `instance` provided in this structure of wasm
        /// exports.
        ///
        /// This function will extract exports from the `instance`
        /// defined within `store` and wrap them all up in the
        /// returned structure which can be used to interact with
        /// the wasm module.
        pub fn new(
            mut store: impl wasmtime::AsContextMut<Data = T>,
            instance: &wasmtime::Instance,
            get_state: impl Fn(&mut T) -> &mut RuneV1Data
                + Send
                + Sync
                + Copy
                + 'static,
        ) -> anyhow::Result<Self> {
            let mut store = store.as_context_mut();
            let start =
                instance.get_typed_func::<(), (), _>(&mut store, "start")?;
            Ok(RuneV1 {
                start,
                get_state: Box::new(get_state),
            })
        }

        /// A function called when the module is first loaded.
        pub fn start(
            &self,
            mut caller: impl wasmtime::AsContextMut<Data = T>,
        ) -> Result<(), wasmtime::Trap> {
            self.start.call(&mut caller, ())?;
            Ok(())
        }
    }
}
pub mod runtime_v1 {
    #[allow(unused_imports)]
    use wit_bindgen_wasmtime::{anyhow, wasmtime};
    #[repr(u8)]
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum TypeHint {
        Integer,
        Float,
        OnelineString,
        MultilineString,
    }
    impl std::fmt::Debug for TypeHint {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                TypeHint::Integer => {
                    f.debug_tuple("TypeHint::Integer").finish()
                },
                TypeHint::Float => f.debug_tuple("TypeHint::Float").finish(),
                TypeHint::OnelineString => {
                    f.debug_tuple("TypeHint::OnelineString").finish()
                },
                TypeHint::MultilineString => {
                    f.debug_tuple("TypeHint::MultilineString").finish()
                },
            }
        }
    }
    /// The various types of values a tensor may contain.
    #[repr(u8)]
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum ElementType {
        Uint8,
        Int8,
        Uint16,
        Int16,
        Uint32,
        Int32,
        Float32,
        Uint64,
        Int64,
        Float64,
    }
    impl std::fmt::Debug for ElementType {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                ElementType::Uint8 => {
                    f.debug_tuple("ElementType::Uint8").finish()
                },
                ElementType::Int8 => {
                    f.debug_tuple("ElementType::Int8").finish()
                },
                ElementType::Uint16 => {
                    f.debug_tuple("ElementType::Uint16").finish()
                },
                ElementType::Int16 => {
                    f.debug_tuple("ElementType::Int16").finish()
                },
                ElementType::Uint32 => {
                    f.debug_tuple("ElementType::Uint32").finish()
                },
                ElementType::Int32 => {
                    f.debug_tuple("ElementType::Int32").finish()
                },
                ElementType::Float32 => {
                    f.debug_tuple("ElementType::Float32").finish()
                },
                ElementType::Uint64 => {
                    f.debug_tuple("ElementType::Uint64").finish()
                },
                ElementType::Int64 => {
                    f.debug_tuple("ElementType::Int64").finish()
                },
                ElementType::Float64 => {
                    f.debug_tuple("ElementType::Float64").finish()
                },
            }
        }
    }
    /// The dimensions that a tensor may have.
    pub enum Dimensions<'a> {
        /// There can be an arbitrary number of dimensions with arbitrary
        /// sizes.
        Dynamic,
        /// The tensor has a fixed rank with the provided dimension sizes.
        ///
        /// If a particular dimension's length is zero, that is interpreted as
        /// the dimension being allowed to have any arbitrary length.
        Fixed(&'a [Le<u32>]),
    }
    impl<'a> std::fmt::Debug for Dimensions<'a> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Dimensions::Dynamic => {
                    f.debug_tuple("Dimensions::Dynamic").finish()
                },
                Dimensions::Fixed(e) => {
                    f.debug_tuple("Dimensions::Fixed").field(e).finish()
                },
            }
        }
    }
    pub trait RuntimeV1: Sized {
        type ArgumentMetadata: std::fmt::Debug;
        type Metadata: std::fmt::Debug;
        type TensorHint: std::fmt::Debug;
        type TensorMetadata: std::fmt::Debug;
        /// Create a new metadata object with the provided name and version
        /// number.
        fn metadata_new(&mut self, name: &str, version: &str)
            -> Self::Metadata;

        /// A human-friendly description of the node.
        ///
        /// The text may include markdown.
        fn metadata_set_description(
            &mut self,
            self_: &Self::Metadata,
            description: &str,
        );

        /// The source repository containing this node.
        fn metadata_set_repository(
            &mut self,
            self_: &Self::Metadata,
            url: &str,
        );

        /// Associate this node with a particular tag.
        ///
        /// Tags are typically used to assist in search and filtering.
        fn metadata_add_tag(&mut self, self_: &Self::Metadata, tag: &str);

        /// Arguments this node accepts.
        fn metadata_add_argument(
            &mut self,
            self_: &Self::Metadata,
            arg: &Self::ArgumentMetadata,
        );

        /// Information about this node's input tensors.
        fn metadata_add_input(
            &mut self,
            self_: &Self::Metadata,
            metadata: &Self::TensorMetadata,
        );

        /// Information about this node's output tensors.
        fn metadata_add_output(
            &mut self,
            self_: &Self::Metadata,
            metadata: &Self::TensorMetadata,
        );

        /// Create a new named argument.
        fn argument_metadata_new(
            &mut self,
            name: &str,
        ) -> Self::ArgumentMetadata;

        /// A human-friendly description of the argument.
        ///
        /// The text may include markdown.
        fn argument_metadata_set_description(
            &mut self,
            self_: &Self::ArgumentMetadata,
            description: &str,
        );

        /// A useful default value for this argument.
        fn argument_metadata_set_default_value(
            &mut self,
            self_: &Self::ArgumentMetadata,
            default_value: &str,
        );

        /// A hint about what type this argument may contain.
        fn argument_metadata_set_type_hint(
            &mut self,
            self_: &Self::ArgumentMetadata,
            hint: TypeHint,
        );

        /// Create a new named tensor.
        fn tensor_metadata_new(&mut self, name: &str) -> Self::TensorMetadata;

        /// A human-friendly description of the tensor.
        ///
        /// The text may include markdown.
        fn tensor_metadata_set_description(
            &mut self,
            self_: &Self::TensorMetadata,
            description: &str,
        );

        /// Add a hint that provides the runtime with contextual information
        /// about this node.
        fn tensor_metadata_add_hint(
            &mut self,
            self_: &Self::TensorMetadata,
            hint: &Self::TensorHint,
        );

        /// Hint to the runtime that a tensor may be displayed as an image.
        fn interpret_as_image(&mut self) -> Self::TensorHint;

        /// Hint to the runtime that a tensor may be interpreted as an audio
        /// clip.
        fn interpret_as_audio(&mut self) -> Self::TensorHint;

        /// Hint that a tensor may have a particular shape and the element types
        /// it supports.
        ///
        /// Note: This hint will be removed in the future in favour of a more
        /// flexible mechanism.
        fn supported_shapes(
            &mut self,
            supported_element_types: Vec<ElementType>,
            dimensions: Dimensions<'_>,
        ) -> Self::TensorHint;

        /// Register a node type with the runtime.
        fn register_node(&mut self, metadata: &Self::Metadata);

        fn drop_argument_metadata(&mut self, state: Self::ArgumentMetadata) {
            drop(state);
        }
        fn drop_metadata(&mut self, state: Self::Metadata) { drop(state); }
        fn drop_tensor_hint(&mut self, state: Self::TensorHint) { drop(state); }
        fn drop_tensor_metadata(&mut self, state: Self::TensorMetadata) {
            drop(state);
        }
    }

    pub struct RuntimeV1Tables<T: RuntimeV1> {
        pub(crate) argument_metadata_table:
            wit_bindgen_wasmtime::Table<T::ArgumentMetadata>,
        pub(crate) metadata_table: wit_bindgen_wasmtime::Table<T::Metadata>,
        pub(crate) tensor_hint_table:
            wit_bindgen_wasmtime::Table<T::TensorHint>,
        pub(crate) tensor_metadata_table:
            wit_bindgen_wasmtime::Table<T::TensorMetadata>,
    }
    impl<T: RuntimeV1> Default for RuntimeV1Tables<T> {
        fn default() -> Self {
            Self {
                argument_metadata_table: Default::default(),
                metadata_table: Default::default(),
                tensor_hint_table: Default::default(),
                tensor_metadata_table: Default::default(),
            }
        }
    }
    pub fn add_to_linker<T, U>(
        linker: &mut wasmtime::Linker<T>,
        get: impl Fn(&mut T) -> (&mut U, &mut RuntimeV1Tables<U>)
            + Send
            + Sync
            + Copy
            + 'static,
    ) -> anyhow::Result<()>
    where
        U: RuntimeV1,
    {
        use wit_bindgen_wasmtime::rt::get_memory;
        linker.func_wrap(
            "runtime-v1",
            "metadata::new",
            move |mut caller: wasmtime::Caller<'_, T>,
                  arg0: i32,
                  arg1: i32,
                  arg2: i32,
                  arg3: i32| {
                let memory = &get_memory(&mut caller, "memory")?;
                let (mem, data) = memory.data_and_store_mut(&mut caller);
                let mut _bc = wit_bindgen_wasmtime::BorrowChecker::new(mem);
                let host = get(data);
                let (host, _tables) = host;
                let ptr0 = arg0;
                let len0 = arg1;
                let ptr1 = arg2;
                let len1 = arg3;
                let param0 = _bc.slice_str(ptr0, len0)?;
                let param1 = _bc.slice_str(ptr1, len1)?;
                let result2 = host.metadata_new(param0, param1);
                Ok(_tables.metadata_table.insert(result2) as i32)
            },
        )?;
        linker.func_wrap(
            "runtime-v1",
            "metadata::set-description",
            move |mut caller: wasmtime::Caller<'_, T>,
                  arg0: i32,
                  arg1: i32,
                  arg2: i32| {
                let memory = &get_memory(&mut caller, "memory")?;
                let (mem, data) = memory.data_and_store_mut(&mut caller);
                let mut _bc = wit_bindgen_wasmtime::BorrowChecker::new(mem);
                let host = get(data);
                let (host, _tables) = host;
                let ptr0 = arg1;
                let len0 = arg2;
                let param0 =
                    _tables.metadata_table.get((arg0) as u32).ok_or_else(
                        || wasmtime::Trap::new("invalid handle index"),
                    )?;
                let param1 = _bc.slice_str(ptr0, len0)?;
                host.metadata_set_description(param0, param1);
                Ok(())
            },
        )?;
        linker.func_wrap(
            "runtime-v1",
            "metadata::set-repository",
            move |mut caller: wasmtime::Caller<'_, T>,
                  arg0: i32,
                  arg1: i32,
                  arg2: i32| {
                let memory = &get_memory(&mut caller, "memory")?;
                let (mem, data) = memory.data_and_store_mut(&mut caller);
                let mut _bc = wit_bindgen_wasmtime::BorrowChecker::new(mem);
                let host = get(data);
                let (host, _tables) = host;
                let ptr0 = arg1;
                let len0 = arg2;
                let param0 =
                    _tables.metadata_table.get((arg0) as u32).ok_or_else(
                        || wasmtime::Trap::new("invalid handle index"),
                    )?;
                let param1 = _bc.slice_str(ptr0, len0)?;
                host.metadata_set_repository(param0, param1);
                Ok(())
            },
        )?;
        linker.func_wrap(
            "runtime-v1",
            "metadata::add-tag",
            move |mut caller: wasmtime::Caller<'_, T>,
                  arg0: i32,
                  arg1: i32,
                  arg2: i32| {
                let memory = &get_memory(&mut caller, "memory")?;
                let (mem, data) = memory.data_and_store_mut(&mut caller);
                let mut _bc = wit_bindgen_wasmtime::BorrowChecker::new(mem);
                let host = get(data);
                let (host, _tables) = host;
                let ptr0 = arg1;
                let len0 = arg2;
                let param0 =
                    _tables.metadata_table.get((arg0) as u32).ok_or_else(
                        || wasmtime::Trap::new("invalid handle index"),
                    )?;
                let param1 = _bc.slice_str(ptr0, len0)?;
                host.metadata_add_tag(param0, param1);
                Ok(())
            },
        )?;
        linker.func_wrap(
            "runtime-v1",
            "metadata::add-argument",
            move |mut caller: wasmtime::Caller<'_, T>, arg0: i32, arg1: i32| {
                let host = get(caller.data_mut());
                let (host, _tables) = host;
                let param0 =
                    _tables.metadata_table.get((arg0) as u32).ok_or_else(
                        || wasmtime::Trap::new("invalid handle index"),
                    )?;
                let param1 = _tables
                    .argument_metadata_table
                    .get((arg1) as u32)
                    .ok_or_else(|| {
                        wasmtime::Trap::new("invalid handle index")
                    })?;
                host.metadata_add_argument(param0, param1);
                Ok(())
            },
        )?;
        linker.func_wrap(
            "runtime-v1",
            "metadata::add-input",
            move |mut caller: wasmtime::Caller<'_, T>, arg0: i32, arg1: i32| {
                let host = get(caller.data_mut());
                let (host, _tables) = host;
                let param0 =
                    _tables.metadata_table.get((arg0) as u32).ok_or_else(
                        || wasmtime::Trap::new("invalid handle index"),
                    )?;
                let param1 = _tables
                    .tensor_metadata_table
                    .get((arg1) as u32)
                    .ok_or_else(|| {
                    wasmtime::Trap::new("invalid handle index")
                })?;
                host.metadata_add_input(param0, param1);
                Ok(())
            },
        )?;
        linker.func_wrap(
            "runtime-v1",
            "metadata::add-output",
            move |mut caller: wasmtime::Caller<'_, T>, arg0: i32, arg1: i32| {
                let host = get(caller.data_mut());
                let (host, _tables) = host;
                let param0 =
                    _tables.metadata_table.get((arg0) as u32).ok_or_else(
                        || wasmtime::Trap::new("invalid handle index"),
                    )?;
                let param1 = _tables
                    .tensor_metadata_table
                    .get((arg1) as u32)
                    .ok_or_else(|| {
                    wasmtime::Trap::new("invalid handle index")
                })?;
                host.metadata_add_output(param0, param1);
                Ok(())
            },
        )?;
        linker.func_wrap(
            "runtime-v1",
            "argument-metadata::new",
            move |mut caller: wasmtime::Caller<'_, T>, arg0: i32, arg1: i32| {
                let memory = &get_memory(&mut caller, "memory")?;
                let (mem, data) = memory.data_and_store_mut(&mut caller);
                let mut _bc = wit_bindgen_wasmtime::BorrowChecker::new(mem);
                let host = get(data);
                let (host, _tables) = host;
                let ptr0 = arg0;
                let len0 = arg1;
                let param0 = _bc.slice_str(ptr0, len0)?;
                let result1 = host.argument_metadata_new(param0);
                Ok(_tables.argument_metadata_table.insert(result1) as i32)
            },
        )?;
        linker.func_wrap(
            "runtime-v1",
            "argument-metadata::set-description",
            move |mut caller: wasmtime::Caller<'_, T>,
                  arg0: i32,
                  arg1: i32,
                  arg2: i32| {
                let memory = &get_memory(&mut caller, "memory")?;
                let (mem, data) = memory.data_and_store_mut(&mut caller);
                let mut _bc = wit_bindgen_wasmtime::BorrowChecker::new(mem);
                let host = get(data);
                let (host, _tables) = host;
                let ptr0 = arg1;
                let len0 = arg2;
                let param0 = _tables
                    .argument_metadata_table
                    .get((arg0) as u32)
                    .ok_or_else(|| {
                        wasmtime::Trap::new("invalid handle index")
                    })?;
                let param1 = _bc.slice_str(ptr0, len0)?;
                host.argument_metadata_set_description(param0, param1);
                Ok(())
            },
        )?;
        linker.func_wrap(
            "runtime-v1",
            "argument-metadata::set-default-value",
            move |mut caller: wasmtime::Caller<'_, T>,
                  arg0: i32,
                  arg1: i32,
                  arg2: i32| {
                let memory = &get_memory(&mut caller, "memory")?;
                let (mem, data) = memory.data_and_store_mut(&mut caller);
                let mut _bc = wit_bindgen_wasmtime::BorrowChecker::new(mem);
                let host = get(data);
                let (host, _tables) = host;
                let ptr0 = arg1;
                let len0 = arg2;
                let param0 = _tables
                    .argument_metadata_table
                    .get((arg0) as u32)
                    .ok_or_else(|| {
                        wasmtime::Trap::new("invalid handle index")
                    })?;
                let param1 = _bc.slice_str(ptr0, len0)?;
                host.argument_metadata_set_default_value(param0, param1);
                Ok(())
            },
        )?;
        linker.func_wrap(
            "runtime-v1",
            "argument-metadata::set-type-hint",
            move |mut caller: wasmtime::Caller<'_, T>, arg0: i32, arg1: i32| {
                let host = get(caller.data_mut());
                let (host, _tables) = host;
                let param0 = _tables
                    .argument_metadata_table
                    .get((arg0) as u32)
                    .ok_or_else(|| {
                        wasmtime::Trap::new("invalid handle index")
                    })?;
                let param1 = match arg1 {
                    0 => TypeHint::Integer,
                    1 => TypeHint::Float,
                    2 => TypeHint::OnelineString,
                    3 => TypeHint::MultilineString,
                    _ => return Err(invalid_variant("TypeHint")),
                };
                host.argument_metadata_set_type_hint(param0, param1);
                Ok(())
            },
        )?;
        linker.func_wrap(
            "runtime-v1",
            "tensor-metadata::new",
            move |mut caller: wasmtime::Caller<'_, T>, arg0: i32, arg1: i32| {
                let memory = &get_memory(&mut caller, "memory")?;
                let (mem, data) = memory.data_and_store_mut(&mut caller);
                let mut _bc = wit_bindgen_wasmtime::BorrowChecker::new(mem);
                let host = get(data);
                let (host, _tables) = host;
                let ptr0 = arg0;
                let len0 = arg1;
                let param0 = _bc.slice_str(ptr0, len0)?;
                let result1 = host.tensor_metadata_new(param0);
                Ok(_tables.tensor_metadata_table.insert(result1) as i32)
            },
        )?;
        linker.func_wrap(
            "runtime-v1",
            "tensor-metadata::set-description",
            move |mut caller: wasmtime::Caller<'_, T>,
                  arg0: i32,
                  arg1: i32,
                  arg2: i32| {
                let memory = &get_memory(&mut caller, "memory")?;
                let (mem, data) = memory.data_and_store_mut(&mut caller);
                let mut _bc = wit_bindgen_wasmtime::BorrowChecker::new(mem);
                let host = get(data);
                let (host, _tables) = host;
                let ptr0 = arg1;
                let len0 = arg2;
                let param0 = _tables
                    .tensor_metadata_table
                    .get((arg0) as u32)
                    .ok_or_else(|| {
                    wasmtime::Trap::new("invalid handle index")
                })?;
                let param1 = _bc.slice_str(ptr0, len0)?;
                host.tensor_metadata_set_description(param0, param1);
                Ok(())
            },
        )?;
        linker.func_wrap(
            "runtime-v1",
            "tensor-metadata::add-hint",
            move |mut caller: wasmtime::Caller<'_, T>, arg0: i32, arg1: i32| {
                let host = get(caller.data_mut());
                let (host, _tables) = host;
                let param0 = _tables
                    .tensor_metadata_table
                    .get((arg0) as u32)
                    .ok_or_else(|| {
                    wasmtime::Trap::new("invalid handle index")
                })?;
                let param1 =
                    _tables.tensor_hint_table.get((arg1) as u32).ok_or_else(
                        || wasmtime::Trap::new("invalid handle index"),
                    )?;
                host.tensor_metadata_add_hint(param0, param1);
                Ok(())
            },
        )?;
        linker.func_wrap(
            "runtime-v1",
            "interpret-as-image",
            move |mut caller: wasmtime::Caller<'_, T>| {
                let host = get(caller.data_mut());
                let (host, _tables) = host;
                let result0 = host.interpret_as_image();
                Ok(_tables.tensor_hint_table.insert(result0) as i32)
            },
        )?;
        linker.func_wrap(
            "runtime-v1",
            "interpret-as-audio",
            move |mut caller: wasmtime::Caller<'_, T>| {
                let host = get(caller.data_mut());
                let (host, _tables) = host;
                let result0 = host.interpret_as_audio();
                Ok(_tables.tensor_hint_table.insert(result0) as i32)
            },
        )?;
        linker.func_wrap(
            "runtime-v1",
            "supported-shapes",
            move |mut caller: wasmtime::Caller<'_, T>,
                  arg0: i32,
                  arg1: i32,
                  arg2: i32,
                  arg3: i32,
                  arg4: i32| {
                let memory = &get_memory(&mut caller, "memory")?;
                let (mem, data) = memory.data_and_store_mut(&mut caller);
                let mut _bc = wit_bindgen_wasmtime::BorrowChecker::new(mem);
                let host = get(data);
                let (host, _tables) = host;
                let len1 = arg1;
                let base1 = arg0;
                let mut result1 = Vec::with_capacity(len1 as usize);
                for i in 0..len1 {
                    let base = base1 + i * 1;
                    result1.push({
                        let load0 = _bc.load::<u8>(base + 0)?;
                        match i32::from(load0) {
                            0 => ElementType::Uint8,
                            1 => ElementType::Int8,
                            2 => ElementType::Uint16,
                            3 => ElementType::Int16,
                            4 => ElementType::Uint32,
                            5 => ElementType::Int32,
                            6 => ElementType::Float32,
                            7 => ElementType::Uint64,
                            8 => ElementType::Int64,
                            9 => ElementType::Float64,
                            _ => return Err(invalid_variant("ElementType")),
                        }
                    });
                }
                let param0 = result1;
                let param1 = match arg2 {
                    0 => Dimensions::Dynamic,
                    1 => Dimensions::Fixed({
                        let ptr2 = arg3;
                        let len2 = arg4;
                        _bc.slice(ptr2, len2)?
                    }),
                    _ => return Err(invalid_variant("Dimensions")),
                };
                let result3 = host.supported_shapes(param0, param1);
                Ok(_tables.tensor_hint_table.insert(result3) as i32)
            },
        )?;
        linker.func_wrap(
            "runtime-v1",
            "register-node",
            move |mut caller: wasmtime::Caller<'_, T>, arg0: i32| {
                let host = get(caller.data_mut());
                let (host, _tables) = host;
                let param0 =
                    _tables.metadata_table.get((arg0) as u32).ok_or_else(
                        || wasmtime::Trap::new("invalid handle index"),
                    )?;
                host.register_node(param0);
                Ok(())
            },
        )?;
        linker.func_wrap(
            "canonical_abi",
            "resource_drop_argument-metadata",
            move |mut caller: wasmtime::Caller<'_, T>, handle: u32| {
                let (host, tables) = get(caller.data_mut());
                let handle = tables
                    .argument_metadata_table
                    .remove(handle)
                    .map_err(|e| {
                        wasmtime::Trap::new(format!(
                            "failed to remove handle: {}",
                            e
                        ))
                    })?;
                host.drop_argument_metadata(handle);
                Ok(())
            },
        )?;
        linker.func_wrap(
            "canonical_abi",
            "resource_drop_metadata",
            move |mut caller: wasmtime::Caller<'_, T>, handle: u32| {
                let (host, tables) = get(caller.data_mut());
                let handle =
                    tables.metadata_table.remove(handle).map_err(|e| {
                        wasmtime::Trap::new(format!(
                            "failed to remove handle: {}",
                            e
                        ))
                    })?;
                host.drop_metadata(handle);
                Ok(())
            },
        )?;
        linker.func_wrap(
            "canonical_abi",
            "resource_drop_tensor-hint",
            move |mut caller: wasmtime::Caller<'_, T>, handle: u32| {
                let (host, tables) = get(caller.data_mut());
                let handle =
                    tables.tensor_hint_table.remove(handle).map_err(|e| {
                        wasmtime::Trap::new(format!(
                            "failed to remove handle: {}",
                            e
                        ))
                    })?;
                host.drop_tensor_hint(handle);
                Ok(())
            },
        )?;
        linker.func_wrap(
            "canonical_abi",
            "resource_drop_tensor-metadata",
            move |mut caller: wasmtime::Caller<'_, T>, handle: u32| {
                let (host, tables) = get(caller.data_mut());
                let handle = tables
                    .tensor_metadata_table
                    .remove(handle)
                    .map_err(|e| {
                        wasmtime::Trap::new(format!(
                            "failed to remove handle: {}",
                            e
                        ))
                    })?;
                host.drop_tensor_metadata(handle);
                Ok(())
            },
        )?;
        Ok(())
    }
    use wit_bindgen_wasmtime::{
        rt::{invalid_variant, RawMem},
        Le,
    };
}
