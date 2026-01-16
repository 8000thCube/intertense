impl<E:Clone> From<&View<E>> for Arc<View<E>>{// TODO viewref to avoid potential cache problems. perhaps give view u8 ptr rather than E ptr and cast according to the generic and a bool
	fn from(value:&View<E>)->Self{value.to_arc()}
}
impl<E:Clone> From<&View<E>> for Box<View<E>>{
	fn from(value:&View<E>)->Self{value.to_box()}
}
impl<E:Clone> ToOwned for View<E>{
	fn to_owned(&self)->Self::Owned{self.to_tensor()}
	type Owned=Tensor<E>;
}
impl<E:Eq> Eq for View<E>{}
impl<E:PartialEq<X>,X> PartialEq<View<X>> for View<E>{
	fn eq(&self,other:&View<X>)->bool{self.dims()==other.dims()&&self.indices().all(|ix|self[&ix]==other[&ix])}
	fn ne(&self,other:&View<X>)->bool{self.dims()!=other.dims()||self.indices().any(|ix|self[&ix]!=other[&ix])}
}
impl<E:RefUnwindSafe> RefUnwindSafe for View<E>{}
impl<E,I:ViewIndex<View<E>>> Index<I> for View<E>{
	fn index(&self,index:I)->&Self::Output{index.index(self)}
	type Output=I::Output;
}
impl<E,I:ViewIndex<View<E>>> IndexMut<I> for View<E>{
	fn index_mut(&mut self,index:I)->&mut Self::Output{index.index_mut(self)}
}
impl<E> AsMut<View<E>> for View<E>{
	fn as_mut(&mut self)->&mut View<E>{self.as_mut_view()}
}
impl<E> AsRef<View<E>> for View<E>{
	fn as_ref(&self)->&View<E>{self.as_view()}
}
impl<E> Default for ViewCache<E>{
	fn default()->Self{
		ViewCache{
			data:[UnsafeCell::new(MaybeUninit::uninit()),UnsafeCell::new(MaybeUninit::uninit()),UnsafeCell::new(MaybeUninit::uninit()),UnsafeCell::new(MaybeUninit::uninit()),UnsafeCell::new(MaybeUninit::uninit()),UnsafeCell::new(MaybeUninit::uninit()),UnsafeCell::new(MaybeUninit::uninit()),UnsafeCell::new(MaybeUninit::uninit())],
			fill:Default::default(),
			next:Mutex::new(None),
			stray:false
		}
	}
}
impl<E> Drop for View<E>{
	fn drop(&mut self){
		if self.stray.is_none(){return}											// no need to do anything if we're not stray. our owner will handle it :3
		unsafe{
			let (ptr,len,cap)=self.stray.unwrap_unchecked();

			if !ptr.is_null(){mem::drop(Vec::from_raw_parts(ptr, len, cap))};	// free the datavec when there isn't a Tensor to do it
			mem::drop(Arc::from_raw(self.views));
		}
	}
}
impl<E> Drop for ViewCache<E>{
	fn drop(&mut self){self.clear()}
}
impl<E> ViewCache<E>{
	/// clears the view cache
	pub (crate) fn clear(&mut self){
		let data=&mut self.data;
		let fill=&mut self.fill;

		for n in 0..fill.load(Ordering::Acquire) as usize{
			if n>=data.len(){break}
			unsafe{data[n].get().as_mut().unwrap_unchecked().assume_init_drop()}
		}
		fill.store(0,Ordering::Release);
	}
	/// checks if the layout is stray
	pub (crate) fn is_stray(&self)->bool{self.stray}
	/// allocates a view in the cache. the cache must last the lifetime of the returned view reference
	pub (crate) unsafe fn alloc<'a>(&self,view:View<E>)->&'a mut View<E>{
		let (data,fill)=(&self.data,&self.fill);
		let l=data.len() as u64;
		let n=fill.fetch_add(1,Ordering::AcqRel);					// hopfully this won't be long running enough to wrap around, but u64 is big enough that should take over a billion years so it's probably fine

		if l<=n{
			return unsafe{
				let mut next=self.next.lock().unwrap();
				if next.is_none(){*next=Some(Default::default())}
				let next=Option::as_ref(&*next).unwrap_unchecked();
				let n=(n as usize)%next.len();

				next[n].alloc(view)
			}
		}
		unsafe{
			let v=data[n as usize].get().as_mut().unwrap_unchecked();

			*v=MaybeUninit::new(view);
			v.assume_init_mut()
		}
	}
	/// creates a new layout ptr that can be used in constructing tensors from raw parts. Do not use the resulting pointer for anything else. Also note that not using it will leak memory. To avoid memory leak and undefined behavior when trying to drop an unused cache ptr, use ViewCache::drop_cache_ptr
	pub fn new_cache_ptr()->*const ViewCache<E>{
		let mut v=ViewCache::default();
		v.stray=true;

		Arc::into_raw(Arc::new(v))
	}
	/// drops the ptr from new_cache_ptr ptr safely
	pub unsafe fn drop_cache_ptr(layout:*const ViewCache<E>){
		let layout=unsafe{Arc::from_raw(layout)};
		assert!(layout.is_stray());
	}
}
impl<E> View<E>{
	// checks if the layout is stray
	//pub (crate) fn is_stray(&self)->bool{self.stray.is_some()}
	/// creates a new stray view
	pub (crate) fn new(mut datavec:Vec<E>,layout:impl Into<Arc<Layout>>)->Self{
		let data=datavec.as_mut_ptr();
		let layout=layout.into();

		assert!(layout.is_valid_for(true,datavec.len()));

		let stray=Some((datavec.as_mut_ptr(),datavec.len(),datavec.capacity()));
		let views=ViewCache::new_cache_ptr();

		Self{data,layout,stray,views}
	}
	/// converts to stray
	pub (crate) fn to_stray(&self)->Self where E:Clone{
		let data=self.data;
		let layout=&self.layout;
		let len=layout.len();

		let dataslice:&[E]=unsafe{slice::from_raw_parts(data,len)};
		let mut datavec=dataslice.to_vec();

		let capacity=datavec.capacity();
		let data=datavec.as_mut_ptr();
		let layout=Arc::as_ptr(layout);
		let ptr=data;
		let viewcache=ViewCache::new_cache_ptr();

		unsafe{Self::from_raw_parts_stray(capacity,data,layout,len,ptr,viewcache)}
	}
	/// creates a view from raw pointers.
	/// data: points to the start of the data buffer
	/// layout: the layout pointer, obtained through Layout::new_layout_ptr, Tensor::layout_ptr, or View::layout_ptr
	/// views: pointer to the view cache, obtained through Tensor::view_cache_ptr, View::view_cache_ptr, or ViewCache::new_cache_ptr
	/// all pointers should be obtained from somewhere with at least the lifetime of the returned reference
	pub (crate) unsafe fn from_raw_parts_stray(capacity:usize,data:*mut E,layout:*const Layout,len:usize,ptr:*mut E,viewcache:*const ViewCache<E>)->Self{
		let data=data as *mut E;
		let layout=unsafe{Arc::from_raw(layout)};
		let stray=Some((ptr, len, capacity));
		let views=unsafe{Arc::from_raw(viewcache)};

		unsafe{
			if !layout.is_stray(){Arc::increment_strong_count(Arc::as_ptr(&layout))}
			if !views.is_stray(){Arc::increment_strong_count(Arc::as_ptr(&views))}
		}

		let views=Arc::into_raw(views);
		//let views=viewcache;
		let view=Self{data,layout,stray,views};

		//unsafe{viewcache.as_ref().unwrap().alloc(view)}
		view
	}
	/// gets the data and layout pointers
	pub fn as_ptr(&self)->(*mut E,*const Layout){
		let data=self.data;
		let layout=Arc::as_ptr(&self.layout);

		(data,layout)
	}
	/// references the view
	pub fn as_view(&self)->&View<E>{self}
	/// references the view
	pub fn as_mut_view(&mut self)->&mut View<E>{self}
	/// expand a dim to dims[n] if it's size 1
	pub fn broadcast<D:AsRef<[usize]>>(&self,dims:D)->&View<E>{
		let dims=dims.as_ref();
		let mut view=self;

		for (d,&n) in dims.iter().enumerate(){
			let d=d as isize;
			view=view.broadcast_dim(d,n);
		}
		view
	}
	/// expand a dim to n if it's size 1
	pub fn broadcast_dim(&self,dim:isize,n:usize)->&View<E>{
		let dim=if let Some(d)=index::normalize_range_stop(self.rank(),dim){d}else{panic!("dim to broadcast did not fit in dims")};
		if self.dims()[dim]==n{return self}

		assert_eq!(self.dims()[dim],1);
		let mut layout=self.layout().clone();

		layout.dims_mut()[dim]=n;
		layout.strides_mut()[dim]=0;	// sharing components is ok here specifically because this is a shared view

		self.with_layout(layout)
	}
	/// points to the cache
	pub fn cache_ptr(&self)->*const ViewCache<E>{self.views}
	/// gets the component at the position. returns none if out of bounds
	pub fn component(&self,indices:&[isize])->Option<&E>{
		let data=self.data;
		let offset=if let Some(o)=self.layout.compute_offset(indices){o}else{return None};

		Some(unsafe{data.add(offset).as_ref().unwrap_unchecked()})
	}
	/// gets the component at the position. returns none if out of bounds
	pub fn component_mut(&mut self,indices:&[isize])->Option<&mut E>{
		let data=self.data;
		let offset=if let Some(o)=self.layout.compute_offset(indices){o}else{return None};

		Some(unsafe{data.add(offset).as_mut().unwrap_unchecked()})
	}
	/// counts the number of components
	pub fn count(&self)->usize{self.layout.count()}
	/// points to the data
	pub fn data_mut_ptr(&mut self)->*mut E{self.data}
	/// points to the data
	pub fn data_ptr(&self)->*const E{self.data as *const E}
	/// references the dimensions
	pub fn dims(&self)->&[usize]{self.layout.dims()}
	/// flattens the view into a vec
	pub fn flat_vec(&self,mem:impl Into<Option<Vec<E>>>)->Vec<E> where E:Clone{
		let dims=self.dims();
		let mut mem=mem.into().unwrap_or_default();
		let mut position=vec![0;self.rank()];

		mem.reserve(self.count());

		index::for_positions(
			dims,
			|ix|mem.extend(self.get_component(ix)),
			&mut position
		);
		mem
	}
	/// flips the view
	pub fn flip(&self)->&View<E>{
		let mut layout=self.layout().clone();

		for stride in layout.strides_mut(){*stride*=-1}
		self.with_layout(layout)
	}
	/// references something by index
	pub fn get<I:ViewIndex<Self>>(&self,index:I)->Option<&I::Output>{index.get(self)}
	/// gets the component at the position. returns none if out of bounds
	pub fn get_component(&self,indices:&[isize])->Option<E> where E:Clone{self.component(indices).cloned()}
	/// references something by index
	pub fn get_mut<I:ViewIndex<Self>>(&mut self,index:I)->Option<&mut I::Output>{index.get_mut(self)}
	/// clones something by index
	pub fn get_owned<I:ViewIndex<Self>>(&self,index:I)->Option<<I::Output as ToOwned>::Owned> where I::Output:ToOwned{index.get(self).map(ToOwned::to_owned)}
	/// returns an iterator over the view indices
	pub fn indices(&self)->GridIter{GridIter::from_shared_layout(self.layout.clone())}
	/// checks if the view is empty
	pub fn is_empty(&self)->bool{self.layout.is_empty()}
	/// checks if the layout is contiguous with lex ordered indices and positive strides, returning true if it is and false if it isn't
	pub fn is_layout_normalized(&self)->bool{self.layout.is_normalized()}
	/// references the layout
	pub fn layout(&self)->&Layout{&self.layout}
	/// returns the layout ptr
	pub fn layout_ptr(&self)->*const Layout{Arc::as_ptr(&self.layout)}
	/// maps the components by reference
	pub fn map<F:FnMut(&E)->Y,Y>(&self,mut f:F)->Tensor<Y>{
		let data=self.indices().map(|ix|f(&self[ix])).collect();
		let dims=self.dims().to_vec();

		Tensor::new(data,dims)
	}
	/// maps the components by reference
	pub fn map_2<F:FnMut(&E,&X)->Y,X,Y>(&self,mut f:F,x:impl AsRef<View<X>>)->Tensor<Y>{
		let x=x.as_ref().broadcast(self.dims());

		let dims=x.dims().to_vec();
		let this=self.broadcast(x.dims());

		let data=self.indices().map(|ix|f(&this[&ix],&x[&ix])).collect();

		Tensor::new(data,dims)
	}
	/// maps the components by reference
	pub fn map_indexed<F:FnMut(&E,Position)->Y,Y>(&self,mut f:F)->Tensor<Y>{
		let data=self.indices().map(|ix|f(&self[&ix],ix)).collect();
		let dims=self.dims().to_vec();

		Tensor::new(data,dims)
	}
	/// creates a new arc view
	pub fn new_arc(data:Vec<E>,dims:Vec<usize>)->Arc<Self>{Arc::new(Self::new(data,Layout::new(dims)))}
	/// creates a new box view
	pub fn new_box(data:Vec<E>,dims:Vec<usize>)->Box<Self>{Box::new(Self::new(data,Layout::new(dims)))}
	/// tests if the views point to the same data in the same layout
	pub fn ptr_eq(&self,other:&Self)->bool{self.data==other.data&&self.layout==other.layout}
	/// returns the number of dims in the tensor
	pub fn rank(&self)->usize{self.layout.rank()}
	/// reshape // TODO try_reshape
	pub fn reshape<D:AsRef<[usize]>>(&self,args:D)->Cow<'_,View<E>> where E:Clone{
		if self.layout.is_normalized(){
			Cow::Borrowed(self.with_layout(Layout::new(args)))
		}else{
			let mut tensor=self.to_owned();

			assert!(tensor.reshape(args));
			Cow::Owned(tensor)
		}
	}
	/// slice
	pub fn slice(&self,index:&[Range<isize>])->Option<&View<E>>{
		let (mut dims,mut strides)=(Vec::new(),Vec::new());
		let mut offset=0;
		let mut viewlayout=Layout::default();

		for ((&d,i),&s) in self.layout.dims().iter().zip(index.iter()).zip(self.layout.strides().iter()){
			let mut n=i.start;
			if s<0{n=!n}																					// negative strides flip the axis, so indices are mapped like -1:0 0:-1 1:-2.., n:-n-1, which is a bit flip on a 2c int

			let start=index::normalize_index(d,n)?;
			let stop=index::normalize_range_stop(d,i.end)?;

			dims.push(stop-start);
			offset+=start*s.abs() as usize;
			strides.push(s);
		}
		*viewlayout.dims_mut()=dims;
		*viewlayout.strides_mut()=strides;

		let data=unsafe{self.data.add(offset)};
		let layout=Arc::new(viewlayout);

		let layout=Arc::as_ptr(&layout);
		let views=self.views;

		Some(unsafe{Self::from_raw_parts(data,layout,views)})
	}
	/// slice mut
	pub fn slice_mut(&mut self,index:&[Range<isize>])->Option<&mut View<E>>{								// code near duplication: copy slice(index) and change mutability
		let (mut dims,mut strides)=(Vec::new(),Vec::new());
		let mut offset=0;
		let mut viewlayout=Layout::default();

		for ((&d,i),&s) in self.layout.dims().iter().zip(index.iter()).zip(self.layout.strides().iter()){
			let mut n=i.start;
			if s<0{n=!n}																					// negative strides flip the axis, so indices are mapped like -1:0 0:-1 1:-2.., n:-n-1, which is a bit flip on a 2c int

			let start=index::normalize_index(d,n)?;
			let stop=index::normalize_range_stop(d,i.end)?;

			dims.push(stop-start);
			offset+=start*s.abs() as usize;
			strides.push(s);
		}
		*viewlayout.dims_mut()=dims;
		*viewlayout.strides_mut()=strides;

		let data=unsafe{self.data.add(offset)};
		let layout=Arc::new(viewlayout);

		let layout=Arc::as_ptr(&layout);
		let views=self.views;

		Some(unsafe{Self::from_raw_parts_mut(data,layout,views)})
	}
	#[track_caller]
	/// remove a dim of size 1
	pub fn squeeze_dim(&self,dim:isize)->&View<E>{
		let dim=if let Some(d)=index::normalize_range_stop(self.rank(),dim){d}else{panic!("dim to unsqueeze did not fit in dims")};
		assert_eq!(self.dims()[dim],1);

		let mut layout=self.layout().clone();

		(layout.dims_mut().remove(dim),layout.strides_mut().remove(dim));
		self.with_layout(layout)
	}
	#[track_caller]
	/// swap dims
	pub fn swap_dims(&self,a:isize,b:isize)->&View<E>{
		let (a,b)=if let (Some(a),Some(b))=(index::normalize_index(self.dims().len(),a),index::normalize_index(self.dims().len(),b)){(a,b)}else{panic!("dims to swap did not fit in dims")};
		if a==b{return self}

		let mut layout=self.layout().clone();
		(layout.dims_mut().swap(a,b),layout.strides_mut().swap(a,b));

		self.with_layout(layout)
	}
	#[track_caller]
	/// swap dims
	pub fn swap_dims_mut(&mut self,a:isize,b:isize)->&mut View<E>{
		let (a,b)=if let (Some(a),Some(b))=(index::normalize_index(self.dims().len(),a),index::normalize_index(self.dims().len(),b)){(a,b)}else{panic!("dims to swap did not fit in dims")};
		if a==b{return self}

		let mut layout=self.layout().clone();
		(layout.dims_mut().swap(a,b),layout.strides_mut().swap(a,b));

		self.with_layout_mut(layout)
	}
	/// converts to an Arc<View>
	pub fn to_arc(&self)->Arc<Self> where E:Clone{self.to_stray().into()}
	/// converts to an Box<View>
	pub fn to_box(&self)->Box<Self> where E:Clone{self.to_stray().into()}
	/// converts to tensor
	pub fn to_tensor(&self)->Tensor<E> where E:Clone{
		let data=self.flat_vec(None);
		let dims=self.dims().to_vec();

		Tensor::new(data,dims)
	}
	/// add a dim of size 1
	pub fn unsqueeze_dim(&self,dim:isize)->&View<E>{
		let dim=if let Some(d)=index::normalize_range_stop(self.rank(),dim){d}else{panic!("dim to unsqueeze did not fit in dims")};
		let mut layout=self.layout().clone();

		(layout.dims_mut().insert(dim,1),layout.strides_mut().insert(dim,1));
		self.with_layout(layout)
	}
	/// returns the view cache pointer
	pub fn view_cache_ptr(&self)->*const ViewCache<E>{self.views}
	/// sets the layout, panicing if it's invalid
	pub fn with_layout(&self,mut layout:Layout)->&Self{
		let data=self.data_ptr();
		let len=self.layout.len();

		assert!(layout.is_valid_for(false,len));

		let layout=Layout::new_layout_ptr(mem::take(layout.dims_mut()),mem::take(layout.strides_mut()));
		let viewcache=self.cache_ptr();

		unsafe{Self::from_raw_parts(data,layout,viewcache)}
	}
	/// sets the layout, panicing if it's invalid
	pub fn with_layout_mut(&mut self,mut layout:Layout)->&mut Self{
		let data=self.data_mut_ptr();
		let len=self.layout.len();

		assert!(layout.is_valid_for(true,len));

		let layout=Layout::new_layout_ptr(mem::take(layout.dims_mut()),mem::take(layout.strides_mut()));
		let viewcache=self.cache_ptr();

		unsafe{Self::from_raw_parts_mut(data,layout,viewcache)}
	}
	/// creates a view from raw pointers.
	/// data: points to the start of the data buffer
	/// layout: the layout pointer, obtained through Layout::new_layout_ptr, Tensor::layout_ptr, or View::layout_ptr. must be valid for the data and its initialized len
	/// views: pointer to the view cache, obtained through Tensor::view_cache_ptr, View::view_cache_ptr, or ViewCache::new_cache_ptr
	/// all pointers should be obtained from somewhere with at least the lifetime of the returned reference
	pub unsafe fn from_raw_parts<'a>(data:*const E,layout:*const Layout,viewcache:*const ViewCache<E>)->&'a Self{
		unsafe{Self::from_raw_parts_mut(data as *mut E,layout,viewcache)}
	}
	/// creates a view from raw pointers.
	/// data: points to the start of the data buffer
	/// layout: the layout pointer, obtained through Layout::new_layout_ptr, Tensor::layout_ptr, or View::layout_ptr. must be valid for the data and its initialized len
	/// views: pointer to the view cache, obtained through Tensor::view_cache_ptr, View::view_cache_ptr, or ViewCache::new_cache_ptr
	/// all pointers should be obtained from somewhere with at least the lifetime of the returned reference
	pub unsafe fn from_raw_parts_mut<'a>(data:*mut E,layout:*const Layout,viewcache:*const ViewCache<E>)->&'a mut Self{
		let data=data as *mut E;
		let layout=unsafe{Arc::from_raw(layout)};
		let stray=None;
		//let views=unsafe{Arc::from_raw(viewcache)};

		unsafe{
			if !layout.is_stray(){Arc::increment_strong_count(Arc::as_ptr(&layout))}
			//if !views.is_stray(){Arc::increment_strong_count(Arc::as_ptr(&views))}
		}

		//let views=Arc::into_raw(views);
		let views=viewcache;
		let view=Self{data,layout,stray,views};

		unsafe{viewcache.as_ref().unwrap().alloc(view)}
		//view
	}
}
#[cfg(test)]
mod tests{
	#[test]
	fn create_stray(){
		let mut datavec=vec![0,1,20,3];
		let layout=Layout::new_layout_ptr(vec![2,2],vec![2,1]);
		let views=ViewCache::new_cache_ptr();

		let capacity=datavec.capacity();
		let data=datavec.as_mut_ptr();
		let len=datavec.len();
		let ptr=data;

		mem::forget(datavec);
		let view=unsafe{View::from_raw_parts_stray(capacity,data,layout,len,ptr,views)};

		let dataslice=unsafe{std::slice::from_raw_parts(view.data,len)};

		assert_eq!(dataslice,[0,1,20,3]);
	}

	use super::*;
}
#[derive(Debug)]
pub struct ViewCache<E>{
	data:[UnsafeCell<MaybeUninit<View<E>>>;8],	// cache views here so they may be referenced. an array prevents soundness problems with shared mutable data in a vec and reduces allocation
	fill:AtomicU64,								// store how full the cache is. increment atomically, don't worry about decrement because that will only be performed withe exclusive access
	next:Mutex<Option<Box<[Self;8]>>>,			// next segments if the cache fills
	stray:bool									// strayness for this struct means the arc containing it will remain allocated without an associated tensor or view
}
#[derive(Debug)]
/// tensor reference type, similar to multidimensional [E]
pub struct View<E>{								// TODO we could implement serial for this.
	data:*mut E,								// buffer or start pointer
	layout:Arc<Layout>,							// arrangement of components, reference counted to avoid having to clone for iterators
	stray:Option<(*mut E,usize,usize)>,			// if the data is not owned by another object, store the vec details here
	views:*const ViewCache<E>					// hack to allow indexing to get a &View even when one doesn't preexist to reference // TODO probably the better way to do this is have the data be *mut union<E,u8>, have the last 8 be a layout pointer and cache layouts since caching the entire view is cursed
}
unsafe impl<E:Send> Send for View<E>{}			// implement send/sync based on E rather than the arena because that holds a cache of pointers rather than the actual data
unsafe impl<E:Sync> Sync for View<E>{}
use std::{
	borrow::{Cow,ToOwned},
	cell::UnsafeCell,
	cmp::{Eq,PartialEq},
	mem::{MaybeUninit,self},
	ops::{Index,IndexMut,Range},
	panic::RefUnwindSafe,
	slice,
	sync::{
		Arc,Mutex,atomic::{AtomicU64,Ordering}
	}
};
use super::{GridIter,Layout,Position,Tensor,ViewIndex,index};
