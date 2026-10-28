//! Statistical trees with separate association-based variable selection and split search.
//!
//! This is an independent, evolving implementation. See `docs/coverage.md` for
//! the distinction between implemented methods and verified GUIDE 46.3 parity.
mod boosting;
mod config;
mod data;
mod discriminant;
mod ensemble;
mod export;
mod importance;
mod node;
mod pruning;
mod quantile;
mod selection;
pub mod stats;
mod tree;

pub use boosting::{BoostedModel, BoostingOptions};
pub use config::*;
pub use data::*;
pub use discriminant::Discriminant;
pub use ensemble::*;
pub use importance::*;
pub use node::{LeafModel, Prediction, PredictionBounds, Term};
pub use pruning::PruningScore;
pub use tree::{FitReport, FittedModel, MissingCodes, Node, SelectionDiagnostic, Split, Tree};

/// Numerical behavior revision recorded in every newly fitted model.
pub const ALGORITHM_VERSION: &str = "guide-rs-0.1-experimental-reference19";

/// Errors from input validation, fitting, prediction, and model serialization.
#[derive(Debug)]
pub struct Error(pub String);
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self(e.to_string())
    }
}
impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self(e.to_string())
    }
}
pub type Result<T> = std::result::Result<T, Error>;

/// Compiles the README's Rust example as a doctest.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error(message.into())
}
