pub enum ConversionError<I,N,O>{Input(I),Intermediate(N),Output(O)}


#[cfg(any(feature="match-tensor",feature="umya-sheet"))]
/// functionality in common with excel and matching related features, named after the xmatch function
mod xmatch;
/// builtin tensor functionality
pub mod builtin_tensor;
#[cfg(feature="burn-ml")]
/// machine learning interop with burn
pub mod burn_ml;
#[cfg(feature="match-tensor")]
/// builtin tensor matching functionality
pub mod match_tensor;
#[cfg(feature="nd-array")]
/// ndarray interop
pub mod nd_array;
#[cfg(feature="serial")]
/// serde as a tensor
pub mod serde_tensor;
#[cfg(feature="umya-sheet")]
/// excel interop with umya spreadsheet
pub mod umya_sheet;


/// converts between two tensor like formats using the builtin tensor as an intermediate
pub fn convert<E,T:Into<Tensor<E>>,U:From<Tensor<E>>>(tensor:T)->U{U::from(tensor.into())}
/// uses conversion to apply the function as if the tensor was from a different library
pub fn inter_map<E,F:FnOnce(U)->U,T:From<Tensor<E>>+Into<Tensor<E>>,U:From<Tensor<E>>+Into<Tensor<E>>>(f:F,x:T)->T{convert(f(convert(x)))}
/// converts between two tensor like formats using the builtin tensor as an intermediate
pub fn try_convert<E,T:TryInto<Tensor<E>>,U:TryFrom<Tensor<E>>>(tensor:T)->Result<U,Either<T::Error,U::Error>>{U::try_from(tensor.try_into().map_err(Either::Left)?).map_err(Either::Right)}
use builtin_tensor::Tensor;
use either::Either;
