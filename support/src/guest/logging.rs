use tracing::{Event, Metadata, Subscriber};
use tracing_subscriber::{
    layer::{Context, SubscriberExt},
    util::SubscriberInitExt,
    Registry,
};

use crate::guest::bindings::{self, LogLevel, LogMetadata, LogValue};

pub(crate) fn initialize_logger() {
    let _ = Registry::default().with(Layer).try_init();
}

struct Layer;

impl<S: Subscriber> tracing_subscriber::Layer<S> for Layer {
    fn enabled(&self, metadata: &Metadata<'_>, _ctx: Context<'_, S>) -> bool {
        bindings::is_enabled(LogMetadata::from(metadata))
    }

    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let meta = LogMetadata::from(event.metadata());

        let mut visitor = Visitor::default();
        event.record(&mut visitor);
        let (msg, data) = visitor.log_values();

        bindings::log(meta, msg, &data);
    }
}

#[derive(Debug)]
enum OwnedLogValue {
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
}

#[derive(Debug, Default)]
struct Visitor(Vec<(&'static str, OwnedLogValue)>);

impl Visitor {
    fn log_values(&self) -> (&str, Vec<(&str, LogValue<'_>)>) {
        let mut values = Vec::new();
        let mut msg = "";

        for (key, value) in &self.0 {
            if let ("message", OwnedLogValue::String(s)) = (*key, value) {
                msg = s.as_str();
                continue;
            }

            let borrowed = match *value {
                OwnedLogValue::Boolean(b) => LogValue::Boolean(b),
                OwnedLogValue::Integer(i) => LogValue::Integer(i),
                OwnedLogValue::Float(f) => LogValue::Float(f),
                OwnedLogValue::String(ref s) => LogValue::String(s),
            };

            values.push((*key, borrowed));
        }

        (msg, values)
    }
}

impl tracing::field::Visit for Visitor {
    fn record_debug(
        &mut self,
        field: &tracing::field::Field,
        value: &dyn std::fmt::Debug,
    ) {
        self.0
            .push((field.name(), OwnedLogValue::String(format!("{value:?}"))));
    }

    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        self.0.push((field.name(), OwnedLogValue::Float(value)));
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.0.push((field.name(), OwnedLogValue::Integer(value)));
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        match i64::try_from(value) {
            Ok(i) => self.0.push((field.name(), OwnedLogValue::Integer(i))),
            Err(_) => self.record_debug(field, &value),
        }
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.0.push((field.name(), OwnedLogValue::Boolean(value)));
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0
            .push((field.name(), OwnedLogValue::String(value.to_string())));
    }

    fn record_error(
        &mut self,
        field: &tracing::field::Field,
        value: &(dyn std::error::Error + 'static),
    ) {
        self.0
            .push((field.name(), OwnedLogValue::String(value.to_string())));

        let mut causes = Vec::new();
        let mut source = value.source();

        while let Some(next_source) = source {
            causes.push(next_source.to_string());
            source = next_source.source();
        }

        if !causes.is_empty() {
            self.0
                .push(("causes", OwnedLogValue::String(format!("{causes:?}"))));
        }
    }
}

impl From<tracing::Level> for LogLevel {
    fn from(level: tracing::Level) -> Self {
        match level {
            tracing::Level::TRACE => LogLevel::Trace,
            tracing::Level::DEBUG => LogLevel::Debug,
            tracing::Level::INFO => LogLevel::Info,
            tracing::Level::WARN => LogLevel::Warn,
            tracing::Level::ERROR => LogLevel::Error,
        }
    }
}

impl<'a> From<&'a Metadata<'a>> for LogMetadata<'a> {
    fn from(metadata: &'a Metadata<'a>) -> Self {
        LogMetadata {
            name: metadata.name(),
            target: metadata.target(),
            level: metadata.level().clone().into(),
            file: metadata.file(),
            line: metadata.line(),
            module: metadata.module_path(),
        }
    }
}
