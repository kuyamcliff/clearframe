//! ClearFrame media layer built on the FFmpeg command-line tools.
pub mod color;
pub mod encode;
pub mod mux;
pub mod pipe;
pub mod probe;
pub mod rational;
pub mod still;
pub mod tools;
pub mod validate;

pub use rational::Rational;
pub use tools::FfmpegTools;

#[derive(Debug, thiserror::Error)]
pub enum MediaError {
    #[error("FFmpeg was not found. Install FFmpeg or set its location in Settings > Advanced.")]
    NotFound,
    #[error("could not start {0}")]
    Spawn(String),
    #[error("{0} timed out")]
    Timeout(String),
    #[error("{0}")]
    Tool(String),
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Process(String),
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
}
