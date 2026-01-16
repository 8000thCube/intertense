impl<I:Display,N:Display,O:Display> Display for ConversionError<I,N,O>{
	fn fmt(&self,f:&mut Formatter<'_>)->FmtResult{
		match self{Self::Input(e)=>e.fmt(f),Self::Intermediate(e)=>e.fmt(f),Self::Output(e)=>e.fmt(f)}
	}
}
impl<I:Error,N:Error,O:Error> Error for ConversionError<I,N,O>{}
#[derive(Clone,Copy,Debug,Eq,Hash,PartialEq)]
pub enum ConversionError<I,N,O>{Input(I),Intermediate(N),Output(O)}


#[cfg(any(feature="match-tensor",feature="umya-sheet"))]
/// functionality in common with excel and matching related features, named after the xmatch function
mod xmatch;
/// builtin tensor functionality
pub mod builtin_tensor;
#[cfg(feature="burn-ml")]
/// machine learning interop with burn
pub mod burn_ml;
#[cfg(feature="image-image")]
pub mod image_image;
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
pub fn try_convert<E,T:TryInto<Tensor<E>>,U:TryFrom<Tensor<E>>>(tensor:T)->Result<U,ConversionError<T::Error,Infallible,U::Error>>{U::try_from(tensor.try_into().map_err(ConversionError::Input)?).map_err(ConversionError::Output)}
/// uses conversion to apply the function as if the tensor was from a different library
pub fn try_inter_map<E,F:FnOnce(U)->Result<U,E>,T:TryFrom<Tensor<X>>+TryInto<Tensor<X>>,U:TryFrom<Tensor<X>>+TryInto<Tensor<X>>,X>(f:F,x:T)->Result<T,ConversionError<ConversionError<<T as TryInto<Tensor<X>>>::Error,Infallible,<U as TryFrom<Tensor<X>>>::Error>,E,ConversionError<<U as TryInto<Tensor<X>>>::Error,Infallible,<T as TryFrom<Tensor<X>>>::Error>>>{
	let intermediate:U=try_convert(x).map_err(ConversionError::Input)?;
	let intermediate:U=f(intermediate).map_err(ConversionError::Intermediate)?;
	let output:T=try_convert(intermediate).map_err(ConversionError::Output)?;

	Ok(output)
}

/// converts between two tensor like formats using the builtin tensor as an intermediate, type specializing some conversions using a typeid based technique. Due to specialization, try from and try into methods may or may not actually be called, and this could unexpectedly succeed in cases where the conversion from/to the builtin tensor type would fail
pub fn try_convert_special<E:Any,T:Any+TryInto<Tensor<E>>,U:Any+TryFrom<Tensor<E>>>(tensor:T)->Result<U,ConversionError<T::Error,Infallible,U::Error>>{
	let (td,ud)=(TypeId::of::<T>(),TypeId::of::<U>());

	if td==ud{		// transmute copy to the same type, forgetting the old one due to manually drop wrapper
		unsafe{return Ok(transmute_move::<T,U>(tensor))}
	}else if td==TypeId::of::<Vec<E>>(){
		/*#[cfg(feature="burn-ml")]// TODO
		if ud==TypeId::of::<BurnTensor<NdArray,1>>{
			unsafe{return Ok(transmute_move::<BurnTensor<NdArray,1>,U>())}
		}*/

		if ud==TypeId::of::<VecDeque<E>>(){
			unsafe{return Ok(transmute_move::<VecDeque<E>,U>(transmute_move::<T,Vec<E>>(tensor).into()))}
		}

	}


	U::try_from(tensor.try_into().map_err(ConversionError::Input)?).map_err(ConversionError::Output)
}


//fn vec_to_tensor_data<E:>()

/// transmute but it works between generic types
unsafe fn transmute_move<T,U>(input:T)->U{
	let input=ManuallyDrop::new(input);
	unsafe{mem::transmute_copy::<ManuallyDrop<T>,U>(&input)}
}

use builtin_tensor::Tensor;
/*#[cfg(feature="burn-ml")]
use burn::{
	backend::{NdArray,Wgpu},prelude::{Tensor as BurnTensor,TensorData}
};*/
use core::{
    any::{Any,TypeId},convert::Infallible,error::Error,fmt::{Display,Formatter,Result as FmtResult},mem::{ManuallyDrop,self},
};
use std::collections::VecDeque;
