impl<C:Deref<Target=[P::Subpixel]>,P:Pixel> From<&ImageBuffer<P,C>> for Tensor<P::Subpixel>{
	fn from(image:&ImageBuffer<P,C>)->Self{
		let channels=P::CHANNEL_COUNT as usize;
		let height=image.height() as usize;
		let width=image.width() as usize;

		let data:Vec<P::Subpixel>=image.pixels().flat_map(|p|p.channels().iter().copied()).collect();
		let dims=vec![height,width,channels];

		assert_eq!(channels*height*width,data.len());
		let mut tensor=Tensor::new(data,dims);

		tensor.swap_dims(0,2);
		tensor.swap_dims(1,2);
		tensor
	}
}
impl<C:Deref<Target=[P::Subpixel]>,P:Pixel> From<ImageBuffer<P,C>> for Tensor<P::Subpixel>{
	fn from(image:ImageBuffer<P,C>)->Self{Self::from(&image)}
}

use crate::builtin_tensor::Tensor;
use image::{ImageBuffer,Pixel};
use std::ops::Deref;



