use std::error;
use std::fmt;

/// Errors that can be returned by RCF operations.
#[derive(Debug, PartialEq)]
pub enum RCFError {
    InvalidArgument {
        msg: &'static str,
    },
}

impl error::Error for RCFError {}

impl fmt::Display for RCFError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            RCFError::InvalidArgument { msg } => write!(f, "{}", msg),
        }
    }
}
