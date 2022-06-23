window.SIDEBAR_ITEMS = {"constant":[["VERSION","Version number of this crate."]],"enum":[["Architecture","The “architecture” field, which in some cases also specifies a specific subarchitecture."],["BinaryFormat","The “binary format” field, which is usually omitted, and the binary format is implied by the other fields."],["CallingConvention","The calling convention, which specifies things like which registers are used for passing arguments, which registers are callee-saved, and so on."],["CompileError","The WebAssembly.CompileError object indicates an error during WebAssembly decoding or validation."],["CompiledFunctionUnwindInfo","Compiled function unwind information."],["CpuFeature","The nomenclature is inspired by the `cpuid` crate. The list of supported features was initially retrieved from `cranelift-native`."],["CustomSectionProtection","Custom section Protection."],["Endianness","The target memory endianness."],["OperatingSystem","The “operating system” field, which sometimes implies an environment, and sometimes isn’t an actual operating system."],["ParseCpuFeatureError","The error that can happen while parsing a `str` to retrieve a `CpuFeature`."],["PointerWidth","The width of a pointer (in the default address space)."],["RelocationKind","Relocation kinds for every ISA."],["RelocationTarget","Destination function. Can be either user function or some special one, like `memory.grow`."],["Symbol","The kinds of wasmer_types objects that might be found in a native object file."],["WasmError","A WebAssembly translation error."]],"fn":[["translate_module","Translate a sequence of bytes forming a valid Wasm binary into a parsed ModuleInfo `ModuleTranslationState`."],["wptype_to_type","Helper function translating wasmparser types to Wasm Type."]],"macro":[["wasm_unsupported","Return an `Err(WasmError::Unsupported(msg))` where `msg` the string built by calling `format!` on the arguments to this macro."]],"struct":[["Compilation","The result of compiling a WebAssembly module’s functions."],["CompileModuleInfo","The required info for compiling a module."],["CompiledFunction","The result of compiling a WebAssembly function."],["CompiledFunctionFrameInfo","The frame info for a Compiled function."],["CustomSection","A Section for a `Compilation`."],["Dwarf","The DWARF information for this Compilation."],["Features","Controls which experimental features will be enabled. Features usually have a corresponding WebAssembly proposal."],["FunctionAddressMap","Function and its instructions addresses mappings."],["FunctionBody","The function body."],["FunctionBodyData","Contains function data: bytecode and its offset in the module."],["InstructionAddressMap","Single source location to generated address mapping."],["MiddlewareBinaryReader","A Middleware binary reader of the WebAssembly structures and types."],["MiddlewareError","A error in the middleware."],["MiddlewareReaderState","The state of the binary reader. Exposed to middlewares to push their outputs."],["ModuleEnvironment","The result of translating via `ModuleEnvironment`. Function bodies are not yet translated, and data initializers have not yet been copied out of the original buffer. The function bodies will be translated by a specific compiler backend."],["ModuleTranslationState","Contains information decoded from the Wasm module that must be referenced during each Wasm function’s translation."],["Relocation","A record of a relocation to perform."],["SectionBody","The bytes in the section."],["SectionIndex","Index type of a Section defined inside a WebAssembly `Compilation`."],["SourceLoc","A source location."],["Target","This is the target that we will use for compiling the WebAssembly ModuleInfo, and then run it."],["TrapInformation","Information about trap."],["Triple","A target “triple”. Historically such things had three fields, though they’ve added additional fields over time."]],"trait":[["Compiler","An implementation of a Compiler from parsed WebAssembly module to Compiled native code."],["CompilerConfig","The compiler configuration options."],["FunctionBinaryReader","Trait for iterating over the operators of a Wasm Function"],["FunctionMiddleware","A function middleware specialized for a single function."],["ModuleMiddleware","A shared builder for function middlewares."],["ModuleMiddlewareChain","Trait for generating middleware chains from “prototype” (generator) chains."],["SymbolRegistry","This trait facilitates symbol name lookups in a native object file."]],"type":[["Addend","Addend to add to the symbol value."],["CodeOffset","Offset in bytes from the beginning of the function."],["CustomSections","The custom sections for a Compilation."],["Functions","The compiled functions map (index in the Wasm -> function)"],["Relocations","Relocations to apply to function bodies."],["WasmResult","A convenient alias for a `Result` that uses `WasmError` as the error type."]]};