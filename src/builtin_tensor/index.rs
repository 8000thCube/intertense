impl AddAssign<&Position> for Position{
	fn add_assign(&mut self,rhs:&Position){
		if rhs.len()>self.len(){
			let lhs=self.clone();

			*self=Position::new(rhs.len());
			*self+=lhs;
			*self+=rhs;
		}
		for (a,b) in self.iter_mut().rev().zip(rhs.iter().rev()){*a+=b}
	}
}
impl AddAssign<Position> for Position{
	fn add_assign(&mut self,rhs:Position){self.add_assign(&rhs)}
}
impl AsMut<[isize]> for Position{
	fn as_mut(&mut self)->&mut [isize]{Arc::make_mut(&mut self.0)}
}
impl AsRef<[isize]> for Position{
	fn as_ref(&self)->&[isize]{self.0.as_ref()}
}
impl AsRef<[usize]> for Position{
	fn as_ref(&self)->&[usize]{
		let ix:&[isize]=self.as_ref();
		unsafe{slice::from_raw_parts(ix.as_ptr() as *const usize,ix.len())}
	}
}
impl Clone for Position{
	fn clone(&self)->Self{Self(self.0.clone())}
	fn clone_from(&mut self,other:&Self){
		if self.len()!=other.len(){return *self=other.clone()}
		Arc::make_mut(&mut self.0).copy_from_slice(other)
	}
}
impl Deref for Position{
	fn deref(&self)->&Self::Target{self.as_ref()}
	type Target=[isize];
}
impl DerefMut for Position{
	fn deref_mut(&mut self)->&mut Self::Target{self.as_mut()}
}
impl DoubleEndedIterator for GridIter{
	fn next_back(&mut self)->Option<Self::Item>{
		if self.start.is_none()&&self.stop.is_none(){			// we use this case to signal nothing left to iterate
			None
		}else if self.stop.is_none(){
			self.stop=self.start.clone();
			self.stop.clone().map(Position)
		}else if self.start.is_none(){							// initial iteration with either start or stop not set: clone one into the other and return it early since we normally preincrement but no increment is required yet. next time they're equal will be time to stop
			self.start=self.stop.clone();
			self.start.clone().map(Position)
		}else{													// increment start as a mixed radix number
			for (&d,n) in self.layout.dims().iter().zip(Arc::make_mut(self.stop.as_mut().unwrap())).rev(){
				*n-=1;
				if *n<0{*n=d as isize-1}else{break}
			}
			if self.start==self.stop{(self.start,self.stop)=(None,None)}
			self.stop.clone().map(Position)						// if stop hasn't been reached yet, start is the current position. otherwise, the iteration is over and we set both start and stop to None
		}
	}
}
impl ExactSizeIterator for GridIter{
	fn len(&self)->usize{
		if self.start.is_none()&&self.stop.is_none(){			// both none case probably not in use but just treat it as nothing to iterate for now
			0
		}else if self.start.is_none()||self.stop.is_none(){		// initial next not called yet. len equal to component count
			component_count(self.layout.dims())
		}else{													// subtract start from stop as a mixed radix subtraction with borrowing
			let mut borrow=0;
			let mut len=0;
			let mut product=1;									// multiply the dimensions to find the place value

			for ((&d,mut start),&stop) in self.layout.dims().iter().zip(self.start.as_ref().unwrap().iter().copied()).zip(self.stop.as_ref().unwrap().iter()).rev(){
				if borrow>0{start+=1}							// add one to the subtractor if this place has been borrowed from to avoid having to handle the subtrahend dropping below 0

				borrow=if start>stop{d}else{0} as isize;		// borrow when this digit of the subtractor is greater than the corresponding digit of the subtrahend
				len+=(borrow+stop-start) as usize*product;		// accumulate the difference to the length
				product*=d;
			}													// since the the index is incremented before returning to save allocations, the difference between stop and start is off by 1. correct by subtracting 1 modulo component count
			if len==0{component_count(self.layout.dims())-1}else{len-1}
		}
	}
}
impl From<Layout> for GridIter{
	fn from(value:Layout)->Self{
		let layout=Arc::new(value);
		let start=if layout.dims().contains(&0){None}else{Some(Arc::from(vec![0;layout.rank()]))};
		let stop=None;

		Self{layout,start,stop}
	}
}
impl From<&[isize]> for Position{
	fn from(value:&[isize])->Self{
		Self(Arc::from(value))
	}
}
impl GridIter{
	/// creates a new indices from the layout
	pub (crate) fn from_shared_layout(layout:Arc<Layout>)->Self{
		let start=if layout.dims().contains(&0){None}else{Some(Arc::from(vec![0;layout.rank()]))};
		let stop=None;

		Self{layout,start,stop}
	}
	/// returns the most recent position from next
	pub fn current(&self)->Option<Position>{self.start.clone().map(Position)}
	// returns the most recent position from next_back
	//pub fn current_back(&self)->Option<Position>{self.stop.clone().map(Position)}
	/// creates a new indices over the specified dimensions. their product should not exceed isize::MAX
	pub fn new<D:AsRef<[usize]>>(dims:D)->Self{
		let layout=Arc::new(Layout::new(dims));
		let start=if layout.dims().contains(&0){None}else{Some(Arc::from(vec![0;layout.rank()]))};
		let stop=None;

		Self{layout,start,stop}
	}
}
impl Iterator for GridIter{
	fn next(&mut self)->Option<Self::Item>{
		if self.start.is_none()&&self.stop.is_none(){			// we use this case to signal nothing left to iterate
			None
		}else if self.start.is_none(){							// initial iteration with either start or stop not set: clone one into the other and return it early since we normally preincrement but no increment is required yet. next time they're equal will be time to stop
			self.start=self.stop.clone();
			self.start.clone().map(Position)
		}else if self.stop.is_none(){
			self.stop=self.start.clone();
			self.stop.clone().map(Position)
		}else{													// increment start as a mixed radix number
			for (&d,n) in self.layout.dims().iter().zip(Arc::make_mut(self.start.as_mut().unwrap())).rev(){
				*n+=1;
				if *n<d as isize{break}else{*n=0}
			}
			if self.start==self.stop{(self.start,self.stop)=(None,None)}
			self.start.clone().map(Position)					// if stop hasn't been reached yet, start is the current position. otherwise, the iteration is over and we set both start and stop to None
		}
	}
	fn nth(&mut self,n:usize)->Option<Self::Item>{
		if n>=self.len(){										// if we advance farther than we have left in the iterator, the iteration is over
			if self.start==self.stop{(self.start,self.stop)=(None,None)}
			None
		}else{													// nth(n) = advance_by(n+1)
			let mut increment=(n+1) as isize;					// increment start as a mixed radix number by n+1, then return it
			if self.start.is_none(){self.start=self.stop.clone()}else if self.stop.is_none(){self.stop=self.start.clone()}

			for (&d,n) in self.layout.dims().iter().zip(Arc::make_mut(self.start.as_mut().unwrap())).rev(){
				let d=d as isize;

				*n+=increment%d;								// add the current digit and shift right
				increment/=d;

				if increment==0&&*n<d{break}					// if all of the increment has been added we can return unless we need to carry
				if *n>=d{*n-=d}									// subtract what carries to the next digit
				increment=1;									// carry
			}
			self.start.clone().map(Position)					// no need to check for having reached stop since that was already cared for in the length check
		}
	}
	fn size_hint(&self)->(usize,Option<usize>){
		let len=self.len();
		(len,Some(len))
	}
	type Item=Position;
}
impl Position{
	/// references as an unsigned coordinates. note that this doesn't perform index normalization, it just casts isize to usize
	pub fn as_unsigned(&self)->&[usize]{self.as_ref()}
	/// creates a new zeroed position
	pub fn new(rank:usize)->Self{Self(vec![0;rank].into())}
}
impl SubAssign<&Position> for Position{
	fn sub_assign(&mut self,rhs:&Position){
		if rhs.len()>self.len(){
			let lhs=self.clone();

			*self=Position::new(rhs.len());
			*self+=lhs;
			*self-=rhs;
		}
		for (a,b) in self.iter_mut().rev().zip(rhs.iter().rev()){*a-=b}
	}
}
impl SubAssign<Position> for Position{
	fn sub_assign(&mut self,rhs:Position){self.sub_assign(&rhs)}
}
impl<const N:usize> From<[isize;N]> for Position{
	fn from(value:[isize;N])->Self{
		Self(Arc::from(value.as_slice()))
	}
}
/// computes the end of a buffer required to hold a tensor with the given dims and strides
pub fn buffer_len(dims:&[usize],strides:&[isize])->usize{
	assert_eq!(dims.len(),strides.len());
	dims.iter().zip(strides).map(|(&d,&s)|(d*s.abs() as usize).wrapping_sub(1)).max().unwrap_or(0).wrapping_add(1)
}
/// counts the components by taking the product of the dims
pub fn component_count(dims:&[usize])->usize{dims.iter().product()}
/// computes the offset of a component
pub fn compute_offset(dims:&[usize],index:&[isize],mut offset:usize,strides:&[isize])->Option<usize>{
	assert_eq!(dims.len(),index.len());
	assert_eq!(dims.len(),strides.len());

	for ((&d,mut n),&s) in dims.iter().zip(index.iter().copied()).zip(strides.iter()){
		if s<0{n=!n}									// negative strides flip the axis, so indices are mapped like -1:0 0:-1 1:-2.., n:-n-1, which is a bit flip on a 2c int
		offset+=normalize_index(d,n)?*s.abs() as usize;
	}
	Some(offset)
}
/// internal iteration over positions that lacks the overhead of the tricks GridIter and Position use to avoid cloning. the iteration will start if the dims don't contain 0. the iteration will stop after reaching a state where each position is one less than the corresponding dim
pub fn for_positions<F:FnMut(&mut [isize])>(dims:&[usize],mut f:F,position:&mut [isize]){
	if dims.contains(&0){return}

	f(position);
	loop{
		if dims.iter().zip(position.iter_mut()).rev().all(|(&n,k)|{
			*k+=1;
			let nextline=n<=*k as usize;
			if nextline{*k=0}
			nextline
		}){
			break
		}else{
			f(position)
		}
	}
}
/// computes the unsigned index given a signed index
pub fn normalize_index(dim:usize,mut index:isize)->Option<usize>{
	if index<0{index+=dim as isize}
	if index<0||index>=dim as isize{return None}
	Some(index as usize)
}
/// computes the unsigned end of range given a signed end of range. unlike normalize_index, when dim==stop, this returns dim rather than none
pub fn normalize_range_stop(dim:usize,stop:isize)->Option<usize>{
	if dim as isize==stop{return Some(dim)}
	normalize_index(dim,stop)
}
#[derive(Clone,Debug,Default,Eq,Hash,PartialEq)]
#[cfg_attr(feature="serial",derive(Deserialize,Serialize))]
/// iterates over a grid of indices
pub struct GridIter{layout:Arc<Layout>,start:Option<Arc<[isize]>>,stop:Option<Arc<[isize]>>}
#[derive(Debug,Default,Eq,Hash,Ord,PartialEq,PartialOrd)]
#[cfg_attr(feature="serial",derive(Deserialize,Serialize))]
#[repr(transparent)]
/// index a tensor by signed position
pub struct Position(Arc<[isize]>);
/// slice index but for tensor views
pub unsafe trait ViewIndex<T:?Sized>{
	/// Returns a shared reference to the output at this location, if in bounds
	fn get(self,view:&T)->Option<&Self::Output>;
	/// Returns a mutable reference to the output at this location, if in bounds.
	fn get_mut(self,view:&mut T)->Option<&mut Self::Output>;
	/// Returns a shared reference to the output at this location, panicking if out of bounds
	fn index(self,view:&T)->&Self::Output;
	/// Returns a mutable reference to the output at this location, panicking if out of bounds
	fn index_mut(self,view:&mut T)->&mut Self::Output;
	/// Returns a pointer to the output at this location, without performing any bounds checking. Calling this method with an out-of-bounds index or a dangling view pointer is undefined behavior even if the resulting pointer is not used.
	unsafe fn get_unchecked(self,view:*const T)->*const Self::Output;
	/// Returns a pointer to the output at this location, without performing any bounds checking. Calling this method with an out-of-bounds index or a dangling view pointer is undefined behavior even if the resulting pointer is not used.
	unsafe fn get_unchecked_mut(self,view:*mut T)->*mut Self::Output;
	/// The output type returned by methods
	type Output:?Sized;
}
unsafe impl<E,const N:usize> ViewIndex<Tensor<E>> for [isize;N]{
	fn get(self,view:&Tensor<E>)->Option<&Self::Output>{ViewIndex::get(self.as_slice(),view)}
	fn get_mut(self,view:&mut Tensor<E>)->Option<&mut Self::Output>{ViewIndex::get_mut(self.as_slice(),view)}
	fn index(self,view:&Tensor<E>)->&Self::Output{ViewIndex::index(self.as_slice(),view)}
	fn index_mut(self,view:&mut Tensor<E>)->&mut Self::Output{ViewIndex::index_mut(self.as_slice(),view)}
	unsafe fn get_unchecked(self,view:*const Tensor<E>)->*const Self::Output{
		unsafe{ViewIndex::get_unchecked(self.as_slice(),view)}
	}
	unsafe fn get_unchecked_mut(self,view:*mut Tensor<E>)->*mut Self::Output{
		unsafe{ViewIndex::get_unchecked_mut(self.as_slice(),view)}
	}
	type Output=E;
}
unsafe impl<E,const N:usize> ViewIndex<View<E>> for [isize;N]{
	fn get(self,view:&View<E>)->Option<&Self::Output>{ViewIndex::get(self.as_slice(),view)}
	fn get_mut(self,view:&mut View<E>)->Option<&mut Self::Output>{ViewIndex::get_mut(self.as_slice(),view)}
	fn index(self,view:&View<E>)->&Self::Output{ViewIndex::index(self.as_slice(),view)}
	fn index_mut(self,view:&mut View<E>)->&mut Self::Output{ViewIndex::index_mut(self.as_slice(),view)}
	unsafe fn get_unchecked(self,view:*const View<E>)->*const Self::Output{
		unsafe{ViewIndex::get_unchecked(self.as_slice(),view)}
	}
	unsafe fn get_unchecked_mut(self,view:*mut View<E>)->*mut Self::Output{
		unsafe{ViewIndex::get_unchecked_mut(self.as_slice(),view)}
	}
	type Output=E;
}
unsafe impl<E> ViewIndex<Tensor<E>> for &[isize]{
	fn get(self,view:&Tensor<E>)->Option<&Self::Output>{view.component(self)}
	fn get_mut(self,view:&mut Tensor<E>)->Option<&mut Self::Output>{view.component_mut(self)}
	fn index(self,view:&Tensor<E>)->&Self::Output{view.component(self).unwrap()}
	fn index_mut(self,view:&mut Tensor<E>)->&mut Self::Output{view.component_mut(self).unwrap()}
	unsafe fn get_unchecked(self,view:*const Tensor<E>)->*const Self::Output{
		let view=unsafe{view.as_ref().unwrap_unchecked()};

		let data=view.data_ptr();
		let offset=view.layout().compute_offset(self).unwrap();

		unsafe{data.add(offset)}
	}
	unsafe fn get_unchecked_mut(self,view:*mut Tensor<E>)->*mut Self::Output{
		let view=unsafe{view.as_mut().unwrap_unchecked()};

		let data=view.data_mut_ptr();
		let offset=view.layout().compute_offset(self).unwrap();

		unsafe{data.add(offset)}
	}
	type Output=E;
}
unsafe impl<E> ViewIndex<Tensor<E>> for &[usize]{
	fn get(self,view:&Tensor<E>)->Option<&Self::Output>{
		if self.iter().any(|&x|x>isize::MAX as usize){return None}
		view.component(unsafe{slice::from_raw_parts(self.as_ptr() as *const isize,self.len())})
	}
	fn get_mut(self,view:&mut Tensor<E>)->Option<&mut Self::Output>{
		if self.iter().any(|&x|x>isize::MAX as usize){return None}
		view.component_mut(unsafe{slice::from_raw_parts(self.as_ptr() as *const isize,self.len())})
	}
	fn index(self,view:&Tensor<E>)->&Self::Output{
		if self.iter().any(|&x|x>isize::MAX as usize){panic!("index greater than usize::MAX")}
		view.component(unsafe{slice::from_raw_parts(self.as_ptr() as *const isize,self.len())}).unwrap()
	}
	fn index_mut(self,view:&mut Tensor<E>)->&mut Self::Output{
		if self.iter().any(|&x|x>isize::MAX as usize){panic!("index greater than usize::MAX")}
		view.component_mut(unsafe{slice::from_raw_parts(self.as_ptr() as *const isize,self.len())}).unwrap()
	}
	unsafe fn get_unchecked(self,view:*const Tensor<E>)->*const Self::Output{
		let view=unsafe{view.as_ref().unwrap_unchecked()};

		let data=view.data_ptr();
		let offset=view.layout().compute_offset(unsafe{slice::from_raw_parts(self.as_ptr() as *const isize,self.len())}).unwrap();

		unsafe{data.add(offset)}
	}
	unsafe fn get_unchecked_mut(self,view:*mut Tensor<E>)->*mut Self::Output{
		let view=unsafe{view.as_mut().unwrap_unchecked()};

		let data=view.data_mut_ptr();
		let offset=view.layout().compute_offset(unsafe{slice::from_raw_parts(self.as_ptr() as *const isize,self.len())}).unwrap();

		unsafe{data.add(offset)}
	}
	type Output=E;
}
unsafe impl<E> ViewIndex<Tensor<E>> for &[Range<isize>]{	// TODO other range types
	fn get(self,view:&Tensor<E>)->Option<&Self::Output>{view.view().slice(self)}
	fn get_mut(self,view:&mut Tensor<E>)->Option<&mut Self::Output>{view.slice_mut(self)}
	fn index(self,view:&Tensor<E>)->&Self::Output{view.view().slice(self).unwrap()}
	fn index_mut(self,view:&mut Tensor<E>)->&mut Self::Output{view.slice_mut(self).unwrap()}
	unsafe fn get_unchecked(self,view:*const Tensor<E>)->*const Self::Output{
		unsafe{view.as_ref().unwrap_unchecked().view().slice(self).unwrap_unchecked() as *const View<E>}
	}
	unsafe fn get_unchecked_mut(self,view:*mut Tensor<E>)->*mut Self::Output{
		unsafe{view.as_mut().unwrap_unchecked().slice_mut(self).unwrap_unchecked() as *mut View<E>}
	}
	type Output=View<E>;
}
unsafe impl<E> ViewIndex<View<E>> for &[isize]{
	fn get(self,view:&View<E>)->Option<&Self::Output>{view.component(self)}
	fn get_mut(self,view:&mut View<E>)->Option<&mut Self::Output>{view.component_mut(self)}
	fn index(self,view:&View<E>)->&Self::Output{view.component(self).unwrap()}
	fn index_mut(self,view:&mut View<E>)->&mut Self::Output{view.component_mut(self).unwrap()}
	unsafe fn get_unchecked(self,view:*const View<E>)->*const Self::Output{
		let view=unsafe{view.as_ref().unwrap_unchecked()};

		let data=view.data_ptr();
		let offset=view.layout().compute_offset(self).unwrap();

		unsafe{data.add(offset)}
	}
	unsafe fn get_unchecked_mut(self,view:*mut View<E>)->*mut Self::Output{
		let view=unsafe{view.as_mut().unwrap_unchecked()};

		let data=view.data_mut_ptr();
		let offset=view.layout().compute_offset(self).unwrap();

		unsafe{data.add(offset)}
	}
	type Output=E;
}
unsafe impl<E> ViewIndex<View<E>> for &Position{
	fn get(self,view:&View<E>)->Option<&Self::Output>{view.component(&self[..])}
	fn get_mut(self,view:&mut View<E>)->Option<&mut Self::Output>{view.component_mut(&self[..])}
	fn index(self,view:&View<E>)->&Self::Output{view.component(&self[..]).unwrap()}
	fn index_mut(self,view:&mut View<E>)->&mut Self::Output{view.component_mut(&self[..]).unwrap()}
	unsafe fn get_unchecked(self,view:*const View<E>)->*const Self::Output{
		unsafe{ViewIndex::get_unchecked(&self[..],view)}
	}
	unsafe fn get_unchecked_mut(self,view:*mut View<E>)->*mut Self::Output{
		unsafe{ViewIndex::get_unchecked_mut(&self[..],view)}
	}
	type Output=E;
}
unsafe impl<E> ViewIndex<Tensor<E>> for Position{
	fn get(self,view:&Tensor<E>)->Option<&Self::Output>{view.component(&self[..])}
	fn get_mut(self,view:&mut Tensor<E>)->Option<&mut Self::Output>{view.component_mut(&self[..])}
	fn index(self,view:&Tensor<E>)->&Self::Output{
		view.component(&self[..]).unwrap_or_else(||panic!("index {self:?} dims {:?}",view.dims()))
	}
	fn index_mut(self,view:&mut Tensor<E>)->&mut Self::Output{view.component_mut(&self[..]).unwrap()}
	unsafe fn get_unchecked(self,view:*const Tensor<E>)->*const Self::Output{
		unsafe{ViewIndex::get_unchecked(&self[..],view)}
	}
	unsafe fn get_unchecked_mut(self,view:*mut Tensor<E>)->*mut Self::Output{
		unsafe{ViewIndex::get_unchecked_mut(&self[..],view)}
	}
	type Output=E;
}
unsafe impl<E> ViewIndex<View<E>> for Position{
	fn get(self,view:&View<E>)->Option<&Self::Output>{view.component(&self[..])}
	fn get_mut(self,view:&mut View<E>)->Option<&mut Self::Output>{view.component_mut(&self[..])}
	fn index(self,view:&View<E>)->&Self::Output{
		view.component(&self[..]).unwrap_or_else(||panic!("index {self:?} dims {:?}",view.dims()))
	}
	fn index_mut(self,view:&mut View<E>)->&mut Self::Output{view.component_mut(&self[..]).unwrap()}
	unsafe fn get_unchecked(self,view:*const View<E>)->*const Self::Output{
		unsafe{ViewIndex::get_unchecked(&self[..],view)}
	}
	unsafe fn get_unchecked_mut(self,view:*mut View<E>)->*mut Self::Output{
		unsafe{ViewIndex::get_unchecked_mut(&self[..],view)}
	}
	type Output=E;
}
unsafe impl<E> ViewIndex<View<E>> for &[usize]{
	fn get(self,view:&View<E>)->Option<&Self::Output>{
		if self.iter().any(|&x|x>isize::MAX as usize){return None}
		view.component(unsafe{slice::from_raw_parts(self.as_ptr() as *const isize,self.len())})
	}
	fn get_mut(self,view:&mut View<E>)->Option<&mut Self::Output>{
		if self.iter().any(|&x|x>isize::MAX as usize){return None}
		view.component_mut(unsafe{slice::from_raw_parts(self.as_ptr() as *const isize,self.len())})
	}
	fn index(self,view:&View<E>)->&Self::Output{
		if self.iter().any(|&x|x>isize::MAX as usize){panic!("index greater than usize::MAX")}
		view.component(unsafe{slice::from_raw_parts(self.as_ptr() as *const isize,self.len())}).unwrap()
	}
	fn index_mut(self,view:&mut View<E>)->&mut Self::Output{
		if self.iter().any(|&x|x>isize::MAX as usize){panic!("index greater than usize::MAX")}
		view.component_mut(unsafe{slice::from_raw_parts(self.as_ptr() as *const isize,self.len())}).unwrap()
	}
	unsafe fn get_unchecked(self,view:*const View<E>)->*const Self::Output{
		let view=unsafe{view.as_ref().unwrap_unchecked()};

		let data=view.data_ptr();
		let offset=view.layout().compute_offset(unsafe{slice::from_raw_parts(self.as_ptr() as *const isize,self.len())}).unwrap();

		unsafe{data.add(offset)}
	}
	unsafe fn get_unchecked_mut(self,view:*mut View<E>)->*mut Self::Output{
		let view=unsafe{view.as_mut().unwrap_unchecked()};

		let data=view.data_mut_ptr();
		let offset=view.layout().compute_offset(unsafe{slice::from_raw_parts(self.as_ptr() as *const isize,self.len())}).unwrap();

		unsafe{data.add(offset)}
	}
	type Output=E;
}
unsafe impl<E> ViewIndex<View<E>> for &[Range<isize>]{	// TODO other range types
	fn get(self,view:&View<E>)->Option<&Self::Output>{view.slice(self)}
	fn get_mut(self,view:&mut View<E>)->Option<&mut Self::Output>{view.slice_mut(self)}
	fn index(self,view:&View<E>)->&Self::Output{view.slice(self).unwrap()}
	fn index_mut(self,view:&mut View<E>)->&mut Self::Output{view.slice_mut(self).unwrap()}
	unsafe fn get_unchecked(self,view:*const View<E>)->*const Self::Output{
		unsafe{view.as_ref().unwrap_unchecked().slice(self).unwrap_unchecked() as *const View<E>}
	}
	unsafe fn get_unchecked_mut(self,view:*mut View<E>)->*mut Self::Output{
		unsafe{view.as_mut().unwrap_unchecked().slice_mut(self).unwrap_unchecked() as *mut View<E>}
	}
	type Output=View<E>;
}
#[cfg(feature="serial")]
use serde::{Deserialize,Serialize};
use std::{
	cmp::PartialOrd,ops::{AddAssign,Deref,DerefMut,Range,SubAssign},slice,sync::Arc
};
use super::{Layout,Tensor,View};
