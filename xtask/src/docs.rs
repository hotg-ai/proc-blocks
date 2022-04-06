use std::io::Write;

use anyhow::Error;
use itertools::Itertools;

use crate::runtime::{
    runtime_v1::ArgumentType, ArgumentHint, ArgumentMetadata, Dimensions,
    Metadata, TensorHint, TensorMetadata,
};

pub fn document(w: &mut dyn Write, meta: &Metadata) -> Result<(), Error> {
    let _span = tracing::info_span!(
        "Generating documentation",
        name = %meta.name,
    )
    .entered();

    let Metadata {
        name,
        version,
        description,
        repository,
        homepage,
        tags,
        arguments,
        inputs,
        outputs,
    } = meta;

    render_title(w, name, version)?;
    render_repo_and_home(w, repository, homepage)?;
    render_tags(w, tags)?;
    render_description(w, description)?;
    render_arguments(w, arguments)?;
    render_tensors(w, "Input Tensors", inputs)?;
    render_tensors(w, "Output Tensors", outputs)?;

    Ok(())
}

fn render_tensors(
    w: &mut dyn Write,
    title: &str,
    outputs: &[TensorMetadata],
) -> Result<(), Error> {
    writeln!(w, "## {title}")?;
    writeln!(w)?;

    if outputs.is_empty() {
        writeln!(w, "*(none)*")?;
        writeln!(w)?;
    }

    for output in outputs {
        render_tensor_metadata(w, output)?;
        writeln!(w)?;
    }

    Ok(())
}

fn render_tensor_metadata(
    w: &mut dyn Write,
    meta: &TensorMetadata,
) -> Result<(), Error> {
    let TensorMetadata {
        name,
        description,
        hints,
    } = meta;

    writeln!(w, "### The `{name}` Tensor")?;
    writeln!(w)?;

    if let Some(description) = description {
        writeln!(w, "{description}")?;
        writeln!(w)?;
    }

    render_tensor_hints(w, hints)?;

    Ok(())
}

fn render_tensor_hints(
    w: &mut dyn Write,
    hints: &Vec<TensorHint>,
) -> Result<(), Error> {
    if hints.is_empty() {
        return Ok(());
    }

    writeln!(w, "Hints:")?;

    for hint in hints {
        render_tensor_hint(w, hint)?;
    }

    Ok(())
}

fn render_tensor_hint(
    w: &mut dyn Write,
    hint: &TensorHint,
) -> Result<(), Error> {
    match hint {
        TensorHint::DisplayAs(ty) => {
            writeln!(w, "- Display as `{ty}`")?;
        },
        TensorHint::SupportedShape {
            accepted_element_types,
            dimensions,
        } => {
            write!(w, "- A ")?;

            match dimensions {
                Dimensions::Dynamic => {
                    write!(w, "dynamically sized tensor")?;
                },
                Dimensions::Fixed(fixed) => {
                    let dims = fixed.iter().join("`, `");
                    writeln!(
                        w,
                        "fixed-size tensor with dimensions [`{dims}`]"
                    )?;
                },
            }

            let elements = accepted_element_types.iter().join("`, `");

            writeln!(w, " which may contain one of `{elements}`")?;
        },
    }

    Ok(())
}

fn render_arguments(
    w: &mut dyn Write,
    arguments: &[ArgumentMetadata],
) -> Result<(), Error> {
    writeln!(w, "## Arguments")?;
    writeln!(w)?;

    if arguments.is_empty() {
        writeln!(w, "*(none)*")?;
        writeln!(w)?;
    }

    for arg in arguments {
        render_argument(w, arg)?;
    }

    Ok(())
}

fn render_argument(
    w: &mut dyn Write,
    arg: &ArgumentMetadata,
) -> Result<(), Error> {
    let ArgumentMetadata {
        name,
        description,
        default_value,
        hints,
    } = arg;

    writeln!(w, "### The `{name}` Argument")?;
    writeln!(w)?;

    if let Some(default) = default_value {
        writeln!(w, "(Default: `{default}`)")?;
        writeln!(w)?;
    }

    if let Some(description) = description {
        writeln!(w, "{description}")?;
        writeln!(w)?;
    }

    render_argument_hints(w, hints)?;

    Ok(())
}

fn render_argument_hints(
    w: &mut dyn Write,
    hints: &[ArgumentHint],
) -> Result<(), Error> {
    if hints.is_empty() {
        return Ok(());
    }

    writeln!(w, "Hints:")?;

    for hint in hints {
        match hint {
            ArgumentHint::NonNegativeNumber => write!(w, "- Non-negative")?,
            ArgumentHint::StringEnum(variants) => {
                let variants = variants.join("`, `");
                writeln!(w, "- One of [`\"{variants}\"`]")?;
            },
            ArgumentHint::NumberInRange { max, min } => {
                writeln!(w, "- A value between `{min}` and `{max}`")?
            },
            ArgumentHint::SupportedArgumentType(ArgumentType::Float) => {
                writeln!(w, "- A float")?
            },
            ArgumentHint::SupportedArgumentType(ArgumentType::Integer) => {
                writeln!(w, "- An integer")?
            },
            ArgumentHint::SupportedArgumentType(
                ArgumentType::UnsignedInteger,
            ) => writeln!(w, "- An unsigned integer")?,
            ArgumentHint::SupportedArgumentType(ArgumentType::String) => {
                writeln!(w, "- A string")?
            },
            ArgumentHint::SupportedArgumentType(ArgumentType::LongString) => {
                writeln!(w, "- A multi-line string")?
            },
        }
    }

    Ok(())
}

fn render_description(
    w: &mut dyn Write,
    description: &Option<String>,
) -> Result<(), Error> {
    if let Some(description) = description {
        writeln!(w, "{description}")?;
        writeln!(w)?;
    }

    Ok(())
}

fn render_title(
    w: &mut dyn Write,
    name: &str,
    version: &str,
) -> Result<(), Error> {
    writeln!(w, "# {name} {version}")?;
    writeln!(w)?;

    Ok(())
}

fn render_repo_and_home(
    w: &mut dyn Write,
    repository: &Option<String>,
    homepage: &Option<String>,
) -> Result<(), Error> {
    match (non_empty(repository), non_empty(homepage)) {
        (None, Some(home)) => {
            writeln!(w, "(*[Homepage]({home})*)")?;
            writeln!(w)?;
        },
        (Some(repo), None) => {
            writeln!(w, "(*[Repository]({repo})*)")?;
            writeln!(w)?;
        },
        (Some(repo), Some(home)) => {
            writeln!(w, "(*[Repository]({repo})|[Homepage]({home})*)")?;
            writeln!(w)?;
        },
        (None, None) => {},
    }

    Ok(())
}

fn render_tags(w: &mut dyn Write, tags: &[String]) -> Result<(), Error> {
    if !tags.is_empty() {
        writeln!(w, "Tags:")?;

        for tag in tags {
            writeln!(w, "- `{}`", tag)?;
        }

        writeln!(w)?;
    }

    Ok(())
}

fn non_empty(value: &Option<impl AsRef<str>>) -> Option<&str> {
    match value.as_ref().map(|v| v.as_ref()) {
        Some(v) if !v.is_empty() => Some(v),
        _ => None,
    }
}
