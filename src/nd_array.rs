impl<A:Clone,D:Dimension> From<ArcArray<A,D>> for Tensor<A>{
	fn from(a:ArcArray<A,D>)->Self{a.into_owned().into()}
}
impl<A,D:Dimension> From<Array<A,D>> for Tensor<A>{
	fn from(a:Array<A,D>)->Self{
		let dims=a.shape().to_vec();
		let mut layout=Layout::new(dims);
		let strides=a.strides().to_vec();

		let (mut data,offset)=a.into_raw_vec_and_offset();
		if let Some(n)=offset{data.truncate(n)}

		*layout.strides_mut()=strides;
		Self::new_with_layout(data,layout)
	}
}
impl<A> From<Tensor<A>> for Array<A,IxDyn>{
	fn from(value:Tensor<A>)->Self{
		let dims=value.dims().to_vec();

		let data=value.flat_vec(None);
		Array::from_shape_vec(IxDyn(&dims),data).unwrap()
	}
}
use crate::builtin_tensor::{Layout,Tensor};
use ndarray::{Array,ArcArray,Dimension,IxDyn};
