/// Given a type parameter `ParamT`, look up its associated type
/// `TypeConstructor::Type` in generic bounds.
pub trait TypeConstructor<ParamT> {
    type Type;
}
