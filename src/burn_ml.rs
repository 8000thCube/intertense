impl<B:Backend,E:Element,K:BasicOps<B>,const N:usize> TryFrom<BuiltinTensor<E>> for Tensor<B,N,K>{// TODO more descriptive error type
	fn try_from(tensor:BuiltinTensor<E>)->Result<Self,Self::Error>{
		if tensor.rank()!=N{return Err(tensor.rank())}
		let dims=tensor.dims().to_vec();

		let data=tensor.flat_vec(None);
		let device=&Default::default();
		let tensordata=TensorData::new(data,dims);

		Ok(Tensor::from_data(tensordata, device))
	}
	type Error=usize;
}
impl<B:Backend,E:Element,K:BasicOps<B>,const N:usize> TryFrom<Tensor<B,N,K>> for BuiltinTensor<E>{
	fn try_from(value:Tensor<B,N,K>)->Result<Self,Self::Error>{
		let data=value.to_data();
		let layout=Layout::new(&data.shape);

		let data=data.convert::<E>().to_vec()?;
		Ok(BuiltinTensor::new_with_layout(data,layout))
	}

	type Error=DataError;
}
#[cfg(feature="serial")]
pub fn deserialize_bool_tensor<'a,B:Backend,D:Deserializer<'a>,const N:usize>(deserializer:D)->Result<Tensor<B,N,Bool>,D::Error>{
	let tensor:BuiltinTensor<bool>=BuiltinTensor::deserialize(deserializer)?;
	tensor.try_into().map_err(|e|Derror::custom(format!("{e:?}")))
}
#[cfg(feature="serial")]
pub fn serialize_bool_tensor<B:Backend,S:Serializer,const N:usize>(tensor:&Tensor<B,N,Bool>,serializer:S)->Result<S::Ok,S::Error>{
	let tensor:BuiltinTensor<bool>=tensor.clone().try_into().map_err(|e|Serror::custom(format!("{e:?}")))?;
	tensor.serialize(serializer)
}
#[cfg(feature="serial")]
pub fn deserialize_float_tensor<'a,B:Backend,D:Deserializer<'a>,const N:usize>(deserializer:D)->Result<Tensor<B,N,Float>,D::Error>{
	let tensor:BuiltinTensor<f32>=BuiltinTensor::deserialize(deserializer)?;
	tensor.try_into().map_err(|e|Derror::custom(format!("{e:?}")))
}
#[cfg(feature="serial")]
pub fn serialize_float_tensor<B:Backend,S:Serializer,const N:usize>(tensor:&Tensor<B,N,Float>,serializer:S)->Result<S::Ok,S::Error>{
	let tensor:BuiltinTensor<f32>=tensor.clone().try_into().map_err(|e|Serror::custom(format!("{e:?}")))?;
	tensor.serialize(serializer)
}
#[cfg(feature="serial")]
pub fn deserialize_int_tensor<'a,B:Backend,D:Deserializer<'a>,const N:usize>(deserializer:D)->Result<Tensor<B,N,Int>,D::Error>{
	let tensor:BuiltinTensor<i32>=BuiltinTensor::deserialize(deserializer)?;
	tensor.try_into().map_err(|e|Derror::custom(format!("{e:?}")))
}
#[cfg(feature="serial")]
pub fn serialize_int_tensor<B:Backend,S:Serializer,const N:usize>(tensor:&Tensor<B,N,Int>,serializer:S)->Result<S::Ok,S::Error>{
	let tensor:BuiltinTensor<i32>=tensor.clone().try_into().map_err(|e|Serror::custom(format!("{e:?}")))?;
	tensor.serialize(serializer)
}
#[cfg(feature="serial")]
pub fn deserialize_uint_tensor<'a,B:Backend,D:Deserializer<'a>,const N:usize>(deserializer:D)->Result<Tensor<B,N,Int>,D::Error>{
	let tensor:BuiltinTensor<u32>=BuiltinTensor::deserialize(deserializer)?;
	tensor.try_into().map_err(|e|Derror::custom(format!("{e:?}")))
}
#[cfg(feature="serial")]
pub fn serialize_uint_tensor<B:Backend,S:Serializer,const N:usize>(tensor:&Tensor<B,N,Int>,serializer:S)->Result<S::Ok,S::Error>{
	let tensor:BuiltinTensor<u32>=tensor.clone().try_into().map_err(|e|Serror::custom(format!("{e:?}")))?;
	tensor.serialize(serializer)
}
use burn::{
	prelude::*,tensor::{BasicOps,DataError,Element}
};
use crate::builtin_tensor::{Layout,Tensor as BuiltinTensor};
#[cfg(feature="serial")]
use serde::{Deserialize,Deserializer,Serialize,Serializer,de::Error as Derror,ser::Error as Serror};

