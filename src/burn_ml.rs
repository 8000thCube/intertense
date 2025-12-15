impl<B:Backend,const N:usize> TryFrom<BuiltinTensor<bool>> for Tensor<B,N,Bool>{
	fn try_from(tensor:BuiltinTensor<bool>)->Result<Self,Self::Error>{
		if tensor.rank()!=N{return Err(tensor.rank())}
		let dims=tensor.dims().to_vec();

		let data=tensor.flat_vec(None);
		let device=&Default::default();
		let tensordata=TensorData::new(data,dims);

		Ok(Tensor::from_data(tensordata, device))
	}
	type Error=usize;
}
impl<B:Backend,const N:usize> TryFrom<BuiltinTensor<f32>> for Tensor<B,N,Float>{
	fn try_from(tensor:BuiltinTensor<f32>)->Result<Self,Self::Error>{
		if tensor.rank()!=N{return Err(tensor.rank())}
		let dims=tensor.dims().to_vec();

		let data=tensor.flat_vec(None);
		let device=&Default::default();
		let tensordata=TensorData::new(data,dims);

		Ok(Tensor::from_data(tensordata, device))
	}
	type Error=usize;
}
impl<B:Backend,const N:usize> TryFrom<BuiltinTensor<i32>> for Tensor<B,N,Int>{
	fn try_from(tensor:BuiltinTensor<i32>)->Result<Self,Self::Error>{
		if tensor.rank()!=N{return Err(tensor.rank())}
		let dims=tensor.dims().to_vec();

		let data=tensor.flat_vec(None);
		let device=&Default::default();
		let tensordata=TensorData::new(data,dims);

		Ok(Tensor::from_data(tensordata, device))
	}
	type Error=usize;
}
impl<B:Backend,const N:usize> TryFrom<BuiltinTensor<u32>> for Tensor<B,N,Int>{
	fn try_from(tensor:BuiltinTensor<u32>)->Result<Self,Self::Error>{
		if tensor.rank()!=N{return Err(tensor.rank())}
		let dims=tensor.dims().to_vec();

		let data=tensor.flat_vec(None);
		let device=&Default::default();
		let tensordata=TensorData::new(data,dims);

		Ok(Tensor::from_data(tensordata, device))
	}
	type Error=usize;
}
impl<B:Backend,const N:usize> TryFrom<Tensor<B,N,Bool>> for BuiltinTensor<bool>{
	fn try_from(value:Tensor<B,N,Bool>)->Result<Self,Self::Error>{
		let data=value.to_data();
		let layout=Layout::new(&data.shape);

		let data=data.to_vec()?;
		Ok(BuiltinTensor::new_with_layout(data,layout))
	}
	type Error=DataError;
}
impl<B:Backend,const N:usize> TryFrom<Tensor<B,N,Float>> for BuiltinTensor<f32>{
	fn try_from(value:Tensor<B,N,Float>)->Result<Self,Self::Error>{
		let data=value.to_data();
		let layout=Layout::new(&data.shape);

		let data=data.to_vec()?;
		Ok(BuiltinTensor::new_with_layout(data,layout))
	}
	type Error=DataError;
}
impl<B:Backend,const N:usize> TryFrom<Tensor<B,N,Int>> for BuiltinTensor<i32>{
	fn try_from(value:Tensor<B,N,Int>)->Result<Self,Self::Error>{
		let data=value.to_data();
		let layout=Layout::new(&data.shape);

		let data=data.to_vec()?;
		Ok(BuiltinTensor::new_with_layout(data,layout))
	}
	type Error=DataError;
}
impl<B:Backend,const N:usize> TryFrom<Tensor<B,N,Int>> for BuiltinTensor<u32>{
	fn try_from(value:Tensor<B,N,Int>)->Result<Self,Self::Error>{
		let data=value.to_data();
		let layout=Layout::new(&data.shape);

		let data=data.to_vec()?;
		Ok(BuiltinTensor::new_with_layout(data,layout))
	}
	type Error=DataError;
}
use burn::{prelude::*,tensor::DataError};
use crate::builtin_tensor::{Layout,Tensor as BuiltinTensor};
