window.SIDEBAR_ITEMS = {"enum":[["ArgumentType","How will an argument be interpreted?"],["DimensionsParam","The dimensions that a tensor may have."],["DimensionsResult","The dimensions that a tensor may have."],["ElementType","The various types of values a tensor may contain."],["LogLevel","The verbosity level used while logging."],["LogValue","A value that can be used when logging structured data."],["ModelInferError",""],["ModelLoadError",""]],"fn":[["ensure_initialized","Make sure all once-off initialization is done."],["interpret_as_audio","Hint to the runtime that a tensor may be interpreted as an audio clip."],["interpret_as_image","Hint to the runtime that a tensor may be displayed as an image."],["interpret_as_number_in_range","Hint to the runtime that an argument may be interpreted as a number in `[min, max]`"],["interpret_as_string_in_enum","Hint to the runtime that an argument may be interpreted as a string in a defined list"],["is_enabled","Check whether a particular message would be logged, allowing the guest to avoid potentially expensive work."],["log","Record a log message with some structured data."],["non_negative_number","Hint to the runtime that an argument may be interpreted as a non-negative number"],["register_node","Register a node type with the runtime."],["supported_argument_type","Tell the runtime that this argument may have a certain type."],["supported_shapes","Hint that a tensor may have a particular shape and the element types it supports."]],"struct":[["ArgumentHint","Hints that can be used by the runtime when inspecting an argument"],["ArgumentMetadata","Information about a node’s argument."],["GraphContext","Contextual information used when determining the ML / Data Processing pipeline. This is defined by the runtime but available for logic within the container (rune)"],["InvalidElementType",""],["KernelContext","Contextual information provided to the guest when evaluating a node."],["LogMetadata","Metadata for a log event."],["Metadata","Metadata describing a single node in the Machine Learning pipeline."],["Model",""],["Shape","The shape of a concrete tensor."],["TensorHint","Hints that can be used by the runtime when inspecting a tensor."],["TensorMetadata","Information about a tensor."],["TensorParam","A tensor with fixed dimensions."],["TensorResult","A tensor with fixed dimensions."]],"type":[["LogValueMap","A list of key-value pairs used when logging structured data."]]};