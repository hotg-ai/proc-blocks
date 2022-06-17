use crate::guest::bindings::*;

impl Metadata {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Metadata {
            name: name.into(),
            version: version.into(),
            tags: Vec::new(),
            description: None,
            homepage: None,
            repository: None,
            arguments: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        let description = description.into();
        if !description.is_empty() {
            self.description = Some(description);
        }

        self
    }

    pub fn with_homepage(mut self, homepage: impl Into<String>) -> Self {
        let homepage = homepage.into();
        if !homepage.is_empty() {
            self.homepage = Some(homepage);
        }

        self
    }

    pub fn with_repository(mut self, repository: impl Into<String>) -> Self {
        let repository = repository.into();
        if !repository.is_empty() {
            self.repository = Some(repository);
        }

        self
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn with_argument(mut self, arg: ArgumentMetadata) -> Self {
        self.arguments.push(arg);
        self
    }

    pub fn with_input(mut self, input: TensorMetadata) -> Self {
        self.inputs.push(input);
        self
    }

    pub fn with_output(mut self, output: TensorMetadata) -> Self {
        self.outputs.push(output);
        self
    }
}

impl ArgumentMetadata {
    pub fn new(name: impl Into<String>) -> Self {
        ArgumentMetadata {
            name: name.into(),
            description: None,
            default_value: None,
            hints: Vec::new(),
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        let description = description.into();
        if !description.is_empty() {
            self.description = Some(description);
        }
        self
    }

    pub fn with_default_value(mut self, default_value: impl ToString) -> Self {
        let default_value = default_value.to_string();
        if !default_value.is_empty() {
            self.default_value = Some(default_value);
        }
        self
    }

    pub fn with_hint(mut self, hint: impl Into<ArgumentHint>) -> Self {
        self.hints.push(hint.into());
        self
    }
}

impl TensorMetadata {
    pub fn new(name: impl Into<String>) -> Self {
        TensorMetadata {
            name: name.into(),
            description: None,
            hints: Vec::new(),
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        let description = description.into();
        if !description.is_empty() {
            self.description = Some(description);
        }

        self
    }

    pub fn with_hint(mut self, hint: TensorHint) -> Self {
        self.hints.push(hint);
        self
    }
}

impl From<MediaType> for TensorHint {
    fn from(m: MediaType) -> Self { TensorHint::MediaType(m) }
}

impl TensorConstraint {
    pub fn new(
        name: impl Into<String>,
        element_type: impl Into<ElementTypeConstraint>,
        dimensions: impl Into<Dimensions>,
    ) -> Self {
        TensorConstraint {
            name: name.into(),
            element_type: element_type.into(),
            dimensions: dimensions.into(),
        }
    }

    pub fn numeric(
        name: impl Into<String>,
        dimensions: impl Into<Dimensions>,
    ) -> Self {
        TensorConstraint {
            name: name.into(),
            element_type: !ElementTypeConstraint::UTF8,
            dimensions: dimensions.into(),
        }
    }
}

impl ArgumentHint {
    pub fn one_of(items: impl IntoIterator<Item = impl ToString>) -> Self {
        ArgumentHint::OneOf(items.into_iter().map(|s| s.to_string()).collect())
    }
}

impl From<ArgumentType> for ArgumentHint {
    fn from(a: ArgumentType) -> Self { ArgumentHint::ArgumentType(a) }
}

impl From<ElementType> for ElementTypeConstraint {
    fn from(e: ElementType) -> Self {
        match e {
            ElementType::U8 => ElementTypeConstraint::U8,
            ElementType::I8 => ElementTypeConstraint::I8,
            ElementType::U16 => ElementTypeConstraint::U16,
            ElementType::I16 => ElementTypeConstraint::I16,
            ElementType::U32 => ElementTypeConstraint::U32,
            ElementType::I32 => ElementTypeConstraint::I32,
            ElementType::F32 => ElementTypeConstraint::F32,
            ElementType::U64 => ElementTypeConstraint::U64,
            ElementType::I64 => ElementTypeConstraint::I64,
            ElementType::F64 => ElementTypeConstraint::F64,
            ElementType::Complex64 => ElementTypeConstraint::COMPLEX64,
            ElementType::Complex128 => ElementTypeConstraint::COMPLEX128,
            ElementType::Utf8 => ElementTypeConstraint::UTF8,
        }
    }
}
