impl<E:Default+DeserializeOwned+Serialize,T:Clone+From<Tensor<E>>+Into<Tensor<E>>> From<T> for SerialTensor<E,T>{
	fn from(value:T)->Self{
		Self{inner:value,phantom:PhantomData}
	}
}
/// deserializes and converts from a tensor
pub fn deserialize_tensor<'a,D:Deserializer<'a>,E,T:From<Tensor<E>>>(deserializer:D)->Result<T,D::Error> where Tensor<E>:Deserialize<'a>{
	let tensor:Tensor<E>=Deserialize::deserialize(deserializer)?;
	Ok(tensor.into())
}
/// clone converts to a tensor and serializes
pub fn serialize_tensor<E,S:Serializer,T:Clone+Into<Tensor<E>>>(data:&T,serializer:S)->Result<S::Ok,S::Error> where Tensor<E>:Serialize{data.clone().into().serialize(serializer)}
/// deserializes and converts from a tensor
pub fn try_deserialize_tensor<'a,D:Deserializer<'a>,E,T:TryFrom<Tensor<E>>>(deserializer:D)->Result<T,D::Error> where T::Error:Display,Tensor<E>:Deserialize<'a>{
	let tensor:Tensor<E>=Deserialize::deserialize(deserializer)?;
	tensor.try_into().map_err(Derror::custom)
}
/// clone converts to a tensor and serializes
pub fn try_serialize_tensor<E,S:Serializer,T:Clone+TryInto<Tensor<E>>>(data:&T,serializer:S)->Result<S::Ok,S::Error> where T::Error:Display,Tensor<E>:Serialize{data.clone().try_into().map_err(Serror::custom)?.serialize(serializer)}
//#[cfg_attr(feature="burn-ml",derive(Module))]// TODO make this implement burn module when inner does
#[derive(Clone,Debug,Deserialize,Serialize)]
#[repr(transparent)]
#[serde(bound="")]
/// wraps a tensor to be serializable by converting to the builtin tensor type
pub struct SerialTensor<E:Default+DeserializeOwned+Serialize,T:Clone+From<Tensor<E>>+Into<Tensor<E>>>{
	#[serde(deserialize_with="deserialize_tensor")]
	#[serde(serialize_with="serialize_tensor")]
	pub inner:T,
	pub phantom:PhantomData<E>,
}
use crate::Tensor;
use serde::{
	Deserialize,Deserializer,Serialize,Serializer,de::{DeserializeOwned,Error as Derror},ser::Error as Serror
};
use std::{fmt::Display,marker::PhantomData};
