//! Reviewed network implementations. Each mirrors the upstream PyTorch module
//! structure so official checkpoints load without renaming (beyond stripping a
//! DataParallel `module.` prefix at install time).
pub mod drunet;
pub mod ffdnet;
pub mod gfpgan;
pub mod nafnet;
pub mod retinaface;
pub mod rife;
pub mod rrdb;
pub mod srvgg;
