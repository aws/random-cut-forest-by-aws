/// Errors that can be returned by RCF operations.
#[derive(Debug, PartialEq)]
pub enum RCFError {
    InvalidArgument {
        msg: &'static str,
    },
}
