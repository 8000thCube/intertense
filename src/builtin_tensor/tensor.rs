fn gcd(a:usize,b:usize)->usize{
	if a==b{return a}

	let (mut a,mut b)=(a.max(b),a.min(b));
	while b>0{(a,b)=(b,a%b)}

	a
}
fn lcm(a:usize,b:usize)->usize{a/gcd(a,b)*b}
impl AsMut<Layout> for Layout{
	fn as_mut(&mut self)->&mut Layout{self}
}
impl AsRef<Layout> for Layout{
	fn as_ref(&self)->&Layout{self}
}
impl From<&[usize]> for Layout{
	fn from(value:&[usize])->Self{Self::new(value)}
}
impl From<Vec<usize>> for Layout{
	fn from(value:Vec<usize>)->Self{Self::new(value)}
}
impl Layout{
	/// checks if the layout is stray
	pub (crate) fn is_stray(&self)->bool{self.stray}
	/// counts the number of components
	pub fn count(&self)->usize{self.dims.iter().product()}
	/// computes the offset of a component given its coordinates. returns none if out of bounds
	pub fn compute_offset(&self,indices:&[isize])->Option<usize>{index::compute_offset(&self.dims,indices,0,&self.strides)}
	/// references the dims
	pub fn dims(&self)->&Vec<usize>{&self.dims}
	/// references the dims. note that most nontrivial functionality expects the lengths of dims and strides to be equal
	pub fn dims_mut(&mut self)->&mut Vec<usize>{&mut self.dims}
	/// converts into the dims
	pub fn into_dims(self)->Vec<usize>{self.dims}
	/// checks if the dims contain 0
	pub fn is_empty(&self)->bool{self.dims.contains(&0)}
	/// computes the length of a slice required to store a tensor with this layout. the backing vec of a tensor will store the data in a slice of this length starting at index offset in a vec of at least offset+len
	pub fn len(&self)->usize{
		let (dims,strides)=(&self.dims,&self.strides);
		let mut acc=1;

		assert_eq!(dims.len(),strides.len());
		for (&d,&s) in dims.iter().zip(strides){
			if d==0{return 0}
			else{acc+=(d-1)*s.abs() as usize}
		}
		return acc;
	}
	/// creates a new layout ptr that can be used in constructing tensors from raw parts. Do not use the resulting pointer for anything else. Also note that not using it will leak memory. To avoid memory leak and undefined behavior when trying to drop an unused layout ptr, use Layout::drop_layout_ptr
	pub fn new_layout_ptr(dims:Vec<usize>,strides:Vec<isize>)->*const Layout{
		let stray=true;
		Arc::into_raw(Arc::new(Layout{dims,stray,strides}))
	}
	/// checks if the layout is contiguous with lex ordered indices and positive strides, returning true if it is and false if it isn't. also returns false if the dims and strides have mismatched lengths, or the product of dims exceeds isize::MAX
	pub fn is_normalized(&self)->bool{
		let mut product:usize=1;
		if self.dims.len()!=self.strides.len(){return false}

		for (&d,&s) in self.dims.iter().zip(self.strides.iter()).rev(){
			let p=product.saturating_mul(d);
			if p>=isize::MAX as usize{return false}

			if s!=product as isize{return false}
			product=p
		}
		true
	}
	/// checks if the layout is valid for storing data in a vector of the specified length (if the vec len is at least self.len, and the product of dims doesn't exceed isize::MAX). also returns false if components would be shared in a mutable case
	pub fn is_valid_for(&self,mutable:bool,veclen:usize)->bool{
		let (dims,strides)=(&self.dims,&self.strides);
		if dims.len()!=strides.len(){return false}

		let saturatingcount=dims.iter().fold(1,|acc:usize,&d|acc.saturating_mul(d));
		if saturatingcount==0{return true}else if saturatingcount>isize::MAX as usize{return false}

		let saturatinglen=dims.iter().zip(strides.iter()).fold(1,|acc:usize,(&d,&s)|acc.saturating_add((d-1).saturating_mul(s.abs() as usize)));
		if saturatinglen>isize::MAX as usize||saturatinglen>veclen{return false}

		if mutable{													// TODO a better way probably exists
			for n in 0..dims.len(){									// overlaps can occur at common multiples between the strides. check if those occur within the axis lengths
				if strides[n]==0{return false}

				for k in 0..n{
					let (d,e)=(dims[n],dims[k]);
					let (s,z)=(strides[n].abs() as usize,strides[k].abs() as usize);

					let (nlen,klen)=(d*s,e*z);
					let maxlen=lcm(s,z);

					if nlen>maxlen&&klen>maxlen{return false}
				}
			}
		}
		true
	}
	/// creates a new tensor layout from the dimensions. their product should not exceed isize::MAX
	pub fn new<D:AsRef<[usize]>>(dims:D)->Self{
		let dims=dims.as_ref().to_vec();
		let mut strides=vec![0;dims.len()];
		let stray=false;

		dims.iter().zip(strides.iter_mut()).rfold(1,|product:usize,(&d,s)|{
			let p=product.saturating_mul(d);
			assert!(p<isize::MAX as usize);

			*s=product as isize;
			p
		});
		Self{dims,stray,strides}
	}
	/// returns the number of dims in the tensor
	pub fn rank(&self)->usize{self.dims.len()}
	/// references the strides
	pub fn strides(&self)->&Vec<isize>{&self.strides}
	/// references the strides. note that most nontrivial functionality expects the lengths of dims and strides to be equal
	pub fn strides_mut(&mut self)->&mut Vec<isize>{&mut self.strides}
	/// drops the ptr from new_layout_ptr ptr safely
	pub unsafe fn drop_layout_ptr(layout:*const Layout){
		let layout=unsafe{Arc::from_raw(layout)};
		assert!(layout.is_stray());
	}
}
impl<E:Clone> From<&[E]> for Tensor<E>{
	fn from(value:&[E])->Self{value.to_vec().into()}
}
impl<E:Clone> From<&View<E>> for Tensor<E>{
	fn from(value:&View<E>)->Self{value.to_tensor()}
}
impl<E:Default> Default for Tensor<E>{
	fn default()->Self{E::default().into()}
}
impl<E:RefUnwindSafe> RefUnwindSafe for Tensor<E>{}
impl<E,I:ViewIndex<Tensor<E>>> Index<I> for Tensor<E>{
	fn index(&self,index:I)->&Self::Output{index.index(self)}
	type Output=I::Output;
}
impl<E,I:ViewIndex<Tensor<E>>> IndexMut<I> for Tensor<E>{
	fn index_mut(&mut self,index:I)->&mut Self::Output{index.index_mut(self)}
}
impl<E> AsMut<Tensor<E>> for Tensor<E>{
	fn as_mut(&mut self)->&mut Tensor<E>{self}
}
impl<E> AsMut<View<E>> for Tensor<E>{
	fn as_mut(&mut self)->&mut View<E>{self.as_mut_view()}
}
impl<E> AsRef<Tensor<E>> for Tensor<E>{
	fn as_ref(&self)->&Tensor<E>{self}
}
impl<E> AsRef<View<E>> for Tensor<E>{
	fn as_ref(&self)->&View<E>{self.as_view()}
}
impl<E> Borrow<View<E>> for Tensor<E>{
	fn borrow(&self)->&View<E>{self.as_view()}
}
impl<E> BorrowMut<View<E>> for Tensor<E>{
	fn borrow_mut(&mut self)->&mut View<E>{self.as_mut_view()}
}
impl<E> Deref for Tensor<E>{
	fn deref(&self)->&Self::Target{self.view()}
	type Target=View<E>;
}
impl<E> DerefMut for Tensor<E>{
	fn deref_mut(&mut self)->&mut Self::Target{self.view_mut()}
}
impl<E> From<E> for Tensor<E>{
	fn from(value:E)->Self{Self::new(vec![value],Vec::new())}
}
impl<E> From<Vec<E>> for Tensor<E>{
	fn from(value:Vec<E>)->Self{
		let dims=vec![value.len()];
		Self::new(value, dims)
	}
}
impl<E> Tensor<E>{
	/// moves the components of other into self, concatenating along the specified axis. the operation may fail due to dimensional mismatch, so it returns false in case of failure
	pub fn append(&mut self,dim:isize,other:&mut Tensor<E>)->bool{
		let (layout,otherlayout)=(&self.layout,&other.layout);
		let rank=layout.rank();

		let dim=if let Some(d)=index::normalize_index(rank, dim){d}else{return false};
		if rank!=otherlayout.rank()|| layout.dims[..dim]!=otherlayout.dims[..dim]||layout.dims[dim+1..]!=otherlayout.dims[dim+1..]{return false}

		let dim=dim as isize;

		assert!(self.swap_dims(dim,0));
		assert!(other.swap_dims(dim,0));

		self.normalize_layout();
		other.normalize_layout();

		let (layout,otherlayout)=(Arc::make_mut(&mut self.layout),Arc::make_mut(&mut other.layout));
		let (len,otherlen)=(layout.dims[0]*layout.strides[0].abs() as usize,otherlayout.dims[0]*otherlayout.strides[0].abs() as usize);

		self.data.truncate(len);

		layout.dims[0]+=mem::take(&mut otherlayout.dims[0]);
		self.data.extend(other.data.drain(..).into_iter().take(otherlen));

		assert!(self.swap_dims(dim,0));
		assert!(other.swap_dims(dim,0));
		other.data.clear();
		true
	}
	/// references the view
	pub fn as_view(&self)->&View<E>{self.view()}
	/// references the view
	pub fn as_mut_view(&mut self)->&mut View<E>{self.view_mut()}
	/// expand a dim to dims[n] if it's size 1
	pub fn broadcast<D:AsRef<[usize]>>(&mut self,dims:D)->bool where E:Clone{
		let dims=dims.as_ref();
		if dims.len()!=self.rank()||dims.iter().zip(self.dims()).any(|(&d,&n)|d!=n&&n!=1){return false}

		for (d,&n) in dims.iter().enumerate(){
			let d=d as isize;
			assert!(self.broadcast_dim(d,n));
		}
		true
	}
	/// expand a dim to n if it's size 1
	pub fn broadcast_dim(&mut self,dim:isize,n:usize)->bool where E:Clone{
		let dim=if let Some(d)=index::normalize_index(self.rank(),dim){d}else{return false};
		if self.dims()[dim]==n{return true}
		if self.dims()[dim]!=1{return false}

		*self=self.view().broadcast_dim(dim as isize,n).to_owned();

		true
	}
	/// points to the cache
	pub fn cache_ptr(&self)->*const ViewCache<E>{Arc::as_ptr(&self.views)}
	/// cat the tensors
	pub fn cat<I:IntoIterator>(dim:isize,tensors:I)->Option<Tensor<E>> where I::Item:Into<Tensor<E>>{
		let mut result:Option<Tensor<E>>=None;

		for tensor in tensors{
			if result.is_none(){result=Some(tensor.into())}else if !result.as_mut().unwrap().append(dim,&mut tensor.into()){return None}
		}
		result
	}
	/// normalizes the layout and removes excess capacity
	pub fn compactify(&mut self){
		let count=self.count();
		self.normalize_layout();

		self.data.truncate(count);
		self.data.shrink_to_fit();
	}
	/// gets the component at the position. returns none if out of bounds
	pub fn component(&self,indices:&[isize])->Option<&E>{
		let offset=if let Some(o)=self.layout.compute_offset(indices){o}else{return None};
		self.data.get(offset)
	}
	/// gets the component at the position. returns none if out of bounds
	pub fn component_mut(&mut self,indices:&[isize])->Option<&mut E>{
		let offset=if let Some(o)=self.layout.compute_offset(indices){o}else{return None};
		self.data.get_mut(offset)
	}
	/// counts the number of components
	pub fn count(&self)->usize{self.layout.count()}
	/// points to the data
	pub fn data_mut_ptr(&mut self)->*mut E{
		assert!(self.is_layout_valid());
		self.views=Default::default();

		self.data.as_mut_ptr()
	}
	/// points to the data
	pub fn data_ptr(&self)->*const E{self.data.as_ptr()}
	/// references the dimensions
	pub fn dims(&self)->&[usize]{self.layout.dims()}
	/// flattens the tensor into a vec
	pub fn flat_vec(mut self,mem:impl Into<Option<Vec<E>>>)->Vec<E>{
		let count=self.count();
		assert!(self.reshape([count]));

		if let Some(mut mem)=mem.into(){
			mem.extend(self.data);
			mem
		}else{
			self.data
		}
	}

	/// flips the dims
	pub fn flip(&mut self)->bool{
		self.views=Default::default();
		let layout=Arc::make_mut(&mut self.layout);

		for stride in layout.strides_mut(){*stride*=-1}
		true
	}
	/// flips the specified tensor dim
	pub fn flip_dim(&mut self,dim:isize)->bool{
		self.views=Default::default();

		let dim=if let Some(d)=index::normalize_range_stop(self.rank(),dim){d}else{return false};
		let layout=Arc::make_mut(&mut self.layout);

		layout.strides_mut()[dim]*=-1;
		true
	}
	/// flips the other dims around the specified axis
	pub fn flip_around(&mut self,axis:isize)->bool{
		self.views=Default::default();

		let axis=if let Some(d)=index::normalize_range_stop(self.rank(),axis){d}else{return false};
		let layout=Arc::make_mut(&mut self.layout);

		layout.strides_mut().iter_mut().enumerate().for_each(|(n,s)|if axis!=n{*s*=-1});
		true
	}
	/// references something by index
	pub fn get<I:ViewIndex<Self>>(&self,index:I)->Option<&I::Output>{index.get(self)}
	/// gets the component at the position. returns none if out of bounds
	pub fn get_component(&self,indices:&[isize])->Option<E> where E:Clone{self.component(indices).cloned()}
	/// gets the tensor layout
	pub fn get_layout(&self)->Layout{self.layout().clone()}
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
	/// checks if the layout is valid
	pub fn is_layout_valid(&self)->bool{self.layout.is_valid_for(true,self.data.len())}
	/// references the layout
	pub fn layout(&self)->&Layout{&self.layout}
	/// references the tensor layout. note that leaving the layout in an invalid state will result in panics or possibly incorrect behavior. set_layout can alleviate this risk by having attempt behavior
	pub fn layout_mut(&mut self)->&mut Layout{Arc::make_mut(&mut self.layout)}
	/// returns the layout ptr
	pub fn layout_ptr(&self)->*const Layout{Arc::as_ptr(&self.layout)}
	/// apply a function to every component
	pub fn map<F:FnMut(E)->Y,Y>(mut self,f:F)->Tensor<Y>{
		self.normalize_layout();

		let count=self.count();
		Tensor::new(
			mem::take(&mut self.data).into_iter().map(f).take(count).collect(),
			self.dims().to_vec()
		)
	}
	/// apply a function to every component. panics if the dims mismatched. this doesn't broadcast, use the view version of the function that works by reference for automatic broadcasting
	pub fn map_2<F:FnMut(E,X)->Y,X,Y>(mut self,mut f:F,mut x:Tensor<X>)->Tensor<Y>{
		assert_eq!(self.dims(),x.dims());

		self.normalize_layout();
		x.normalize_layout();

		let count=self.count();
		Tensor::new(
			mem::take(&mut self.data).into_iter().zip(x.data.drain(..))
				.map(|(e,x)|f(e,x)).take(count).collect(),
			self.dims().to_vec()
		)
	}
	/// apply a function to every component
	pub fn map_indexed<F:FnMut(E,Position)->Y,Y>(mut self,mut f:F)->Tensor<Y>{
		let indices=self.indices();
		self.normalize_layout();

		let count=self.count();
		Tensor::new(
			mem::take(&mut self.data).into_iter().zip(indices)
				.map(|(e,ix)|f(e,ix)).take(count).collect(),
			self.dims().to_vec()
		)
	}
	/// creates a new tensor from the data and dimensions
	pub fn new(data:Vec<E>,dims:Vec<usize>)->Self{
		Self{
			data,
			layout:Layout::new(dims).into(),
			views:Default::default()
		}
	}
	/// creates a new tensor from the dimensions, and default values for the components
	pub fn new_with<F:FnMut()->E>(datagen:F,layout:impl Into<Layout>)->Self{
		let mut data=Vec::new();
		let layout=layout.into();

		data.resize_with(layout.len(),datagen);
		Self::new_with_layout(data,layout)
	}
	/// creates a new tensor from the data and dimensions. layout is not checked, but attempting operations with an invalid layout will likely result in incorrect behavior or panics
	pub fn new_with_layout(data:Vec<E>,layout:Layout)->Self{
		Self{
			data,
			layout:layout.into(),
			views:Default::default()
		}
	}
	/// rearranges the components in memory and layout so the layout is normalized
	pub fn normalize_layout(&mut self){
		//assert!(self.is_layout_valid());
		self.views=Default::default();

		if self.is_layout_normalized(){return}

		let count=self.count();
		let dims=&self.layout.dims;
		let indices=self.indices();
		let len=self.layout.len();
		let newlayout=Layout::new(dims);
		let strides=&self.layout.strides;

		let data:Vec<MaybeUninit<E>>=mem::take(&mut self.data).into_iter().map(MaybeUninit::new).collect();
		let mut forgetmask=vec![false;len];

		self.data.reserve(count);
		unsafe{
			for ix in indices{
				let offset=index::compute_offset(dims,&ix,0,strides).unwrap();

				assert!(!forgetmask[offset],"validity check should include no overlap");
				forgetmask[offset]=true;
				self.data.push(data[offset].assume_init_read());
			}
			for mut deleted in data.into_iter().zip(forgetmask).filter_map(|(data,forget)|(!forget).then_some(data)){
				deleted.assume_init_drop();
			}
		}

		self.layout=newlayout.into();
	}
	/// pads the dimension to size with val. returns false if out of bounds. returns true but does nothing if the dimensions size is greater than size
	pub fn pad_dim(&mut self,dim:isize,size:usize,val:E)->bool where E:Clone{self.pad_dim_with(dim,size,||val.clone())}
	/// pads the dimension to size with val. returns false if out of bounds. returns true but does nothing if the dimensions size is greater than size
	pub fn pad_dim_with<F:FnMut()->E>(&mut self,dim:isize,size:usize,val:F)->bool{
		let dim=if let Some(d)=index::normalize_index(self.dims().len(),dim){d}else{return false} as isize;
		if self.dims()[dim as usize]>=size{return true}

		assert!(self.swap_dims(dim,0));
		self.normalize_layout();

		let layout=Arc::make_mut(&mut self.layout);
		let selfbuf=&mut self.data;

		layout.dims[0]=size;
		selfbuf.resize_with(size*layout.strides()[0].abs() as usize,val);

		assert!(self.swap_dims(dim,0));
		true
	}
	/// mtaches the dims of same rank tensors by padding
	pub fn pad_match<'a,A:IntoIterator<Item=&'a mut I>,I:'a+AsMut<Tensor<E>>>(tensors:A,val:E)->bool where E:Clone{Self::pad_match_with(tensors,||val.clone())}
	/// mtaches the dims of same rank tensors by padding
	pub fn pad_match_with<'a,A:IntoIterator<Item=&'a mut I>,F:FnMut()->E,I:'a+AsMut<Tensor<E>>>(tensors:A,mut val:F)->bool{
		let mut tensors:Vec<&'a mut I>=tensors.into_iter().collect();
		if tensors.len()==0{return true}

		let mut dims=tensors[0].as_mut().dims().to_vec();

		for d in tensors.iter_mut().map(|i|i.as_mut().dims()){
			if d.len()!=dims.len(){return false}
			for (&d,e) in d.iter().zip(dims.iter_mut()){*e=d.max(*e)}
		}
		for t in tensors.iter_mut().map(|i|i.as_mut()){
			for d in 0..dims.len(){
				if !t.pad_dim_with(d as isize,dims[d],&mut val){return false}
			}
		}
		true
	}
	/// returns the number of dims in the tensor
	pub fn rank(&self)->usize{self.layout.rank()}
	/// rearranges the components and sets the dimensions such that the components' positions have the same lex order, but the dimensions are as specified. the function succeeds if the product of the current dimensions equals the product of the new dimensions. returns false if failure
	pub fn reshape<D:AsRef<[usize]>>(&mut self,d:D)->bool{
		assert!(self.is_layout_valid());
		self.views=Default::default();

		let newlayout=Layout::new(d);
		if newlayout.count()!=self.count(){return false}

		self.normalize_layout();
		self.set_layout(newlayout)
	}
	/// sets the layout if it's valid
	pub fn set_layout(&mut self,layout:Layout)->bool{
		self.views=Default::default();

		if !layout.is_valid_for(true,self.data.len()){return false}
		self.layout=layout.into();

		return true
	}
	/// slices the tensor in place
	pub fn slice(&mut self,ranges:&[Range<isize>])->bool{// TODO slice_dim, convenience/trait: x_in_place/with_x, lazy: xed, tensor/view: x/view_x
		//assert!(self.is_layout_valid());
		self.views=Default::default();

		let (mut dims,mut strides)=(Vec::new(),Vec::new());
		let mut offset=0;
		let mut viewlayout=Layout::default();

		for ((&d,i),&s) in self.layout.dims().iter().zip(ranges.iter()).zip(self.layout.strides().iter()){
			let mut n=i.start;
			if s<0{n=!n}																					// negative strides flip the axis, so indices are mapped like -1:0 0:-1 1:-2.., n:-n-1, which is a bit flip on a 2c int

			let start=if let Some(s)=index::normalize_index(d,n){s}else{return false};
			let stop=if let Some(s)=index::normalize_range_stop(d,i.end){s}else{return false};

			dims.push(stop-start);
			offset+=start*s.abs() as usize;
			strides.push(s);
		}
		*viewlayout.dims_mut()=dims;
		*viewlayout.strides_mut()=strides;

		self.data.drain(..offset);
		self.data.truncate(viewlayout.len());
		self.layout=viewlayout.into();

		true
	}
	/// stack the tensors
	pub fn stack<I:IntoIterator>(dim:isize,tensors:I)->Option<Tensor<E>> where I::Item:Into<Tensor<E>>{
		let mut result:Option<Tensor<E>>=None;

		for tensor in tensors{
			let mut tensor=tensor.into();

			if !tensor.unsqueeze_dim(dim){return None}
			if result.is_none(){result=Some(tensor.into())}else if !result.as_mut().unwrap().append(dim,&mut tensor){return None}
		}
		result
	}
	/// remove a dim if it's size 1
	pub fn squeeze_dim(&mut self,dim:isize)->bool{
		self.views=Default::default();

		let dim=if let Some(d)=index::normalize_index(self.rank(),dim){d}else{return false};
		if self.dims()[dim]!=1{return false}

		let layout=Arc::make_mut(&mut self.layout);

		layout.dims.remove(dim);
		layout.strides.remove(dim);
		true
	}
	/// swap dims
	pub fn swap_dims(&mut self,a:isize,b:isize)->bool{
		self.views=Default::default();

		let (a,b)=if let (Some(a),Some(b))=(index::normalize_index(self.dims().len(),a),index::normalize_index(self.dims().len(),b)){(a,b)}else{return false};
		if a==b{return true}

		let layout=Arc::make_mut(&mut self.layout);
		(layout.dims.swap(a,b),layout.strides.swap(a,b));

		true
	}
	/// add a dim of size 1
	pub fn unsqueeze_dim(&mut self,dim:isize)->bool{
		let dim=if let Some(d)=index::normalize_range_stop(self.rank(),dim){d}else{return false};
		self.views=Default::default();

		let layout=Arc::make_mut(&mut self.layout);

		layout.dims.insert(dim,1);
		layout.strides.insert(dim,1);

		true
	}
	/// rereferences as a view
	pub fn view(&self)->&View<E>{
		//assert!(self.is_layout_valid());
		assert!(self.layout.is_valid_for(false,self.data.len()));	// TODO it's probably okay if someone makes a shared component layout only to take a shared view. document or undo before stabilizing.

		let data=self.data_ptr();
		let layout=self.layout_ptr();
		let views=self.view_cache_ptr();

		unsafe{View::from_raw_parts(data,layout,views)}
	}
	/// rereferences as a view
	pub fn view_mut(&mut self)->&mut View<E>{
		assert!(self.is_layout_valid());
		self.views=Default::default();

		let data=self.data_mut_ptr();
		let layout=self.layout_ptr();
		let views=self.view_cache_ptr();

		unsafe{View::from_raw_parts_mut(data,layout,views)}
	}
	/// returns the view cache pointer
	pub fn view_cache_ptr(&self)->*const ViewCache<E>{Arc::as_ptr(&self.views)}
	/// creates tensor from raw parts:
	/// capacity: capacity of the underlying buffer
	/// data: pointer to the buffer
	/// layout: the layout pointer, obtained through Layout::new_layout_ptr, Tensor::layout_ptr, or View::layout_ptr
	/// view: pointer to the view cache, obtained through Tensor::view_cache_ptr, View::view_cache_ptr, or ViewCache::new_cache_ptr
	pub unsafe fn from_raw_parts(capacity:usize,data:*mut E,layout:*const Layout,len:usize,viewcache:*const ViewCache<E>)->Self{
		let data=unsafe{Vec::from_raw_parts(data,capacity,len)};
		let layout=unsafe{Arc::from_raw(layout)};
		let views=unsafe{Arc::from_raw(viewcache)};

		unsafe{
			if !layout.is_stray(){Arc::increment_strong_count(Arc::as_ptr(&layout))}
			if !views.is_stray(){Arc::increment_strong_count(Arc::as_ptr(&views))}
		}

		Self{data,layout,views}
	}
}
#[cfg(test)]
mod tests{
	#[test]
	fn indices_1x4t2x2(){
		let tensor:Tensor<char>=Tensor::new(vec!['a','b','c','d'],vec![4]);

		let mut indices=tensor.view().reshape(&[2,2]).indices();

		assert_eq!(indices.len(),4);
		assert_eq!(*indices.next().unwrap(),[0,0]);
		assert_eq!(indices.len(),3);
		assert_eq!(*indices.next().unwrap(),[0,1]);
		assert_eq!(indices.len(),2);
		assert_eq!(*indices.next().unwrap(),[1,0]);
		assert_eq!(indices.len(),1);
		assert_eq!(*indices.next().unwrap(),[1,1]);
		assert_eq!(indices.len(),0);
		assert_eq!(indices.next(),None);
		assert_eq!(indices.len(),0);
		assert_eq!(indices.next(),None);
	}
	#[test]
	fn make_and_view_vector(){
		let data:Vec<i32>=vec![-1,-2,0,1,5];
		let tensor:Tensor<i32>=data.into();

		assert_eq!(tensor.dims(),[5]);
		assert_eq!(tensor[[0_isize]],-1);
		assert_eq!(tensor[[1_isize]],-2);
		assert_eq!(tensor[[2_isize]],0);
		assert_eq!(tensor[[3_isize]],1);
		assert_eq!(tensor[[4_isize]],5);

		let view=&tensor[[1..4].as_slice()];
		assert_eq!(view[[0_isize]],-2);
		assert_eq!(view[[1_isize]],0);
		assert_eq!(view[[2_isize]],1);
	}
	#[test]
	fn reshape_1x4t2x2(){
		let tensor:Tensor<char>=Tensor::new(vec!['a','b','c','d'],vec![4]);
		let view=tensor.view().reshape([2,2]);

		assert_eq!(view.dims(),[2,2]);
		assert_eq!(view[[0,0]],'a');
		assert_eq!(view[[1,0]],'c');
		assert_eq!(view[[0,1]],'b');
		assert_eq!(view[[1,1]],'d');

	}
	#[test]
	fn broadcast_00(){
		let input:Vec<i32>=vec![1,2,3,4];
		let template:Vec<i32>=vec![1,1,1,2,2,2,3,3,3,4,4,4];

		let mut response:Tensor<i32>=Tensor::from(input.as_slice());

		assert!(response.unsqueeze_dim(1));
		assert!(response.broadcast_dim(1,3));
		assert_eq!(response.flat_vec(None),template);

		let response=Tensor::<i32>::from(input).view().unsqueeze_dim(1).broadcast_dim(1,3).to_owned();
		assert_eq!(response.flat_vec(None),template);
	}
	use super::*;
}
#[cfg_attr(feature="serial",derive(Deserialize,Serialize))]
#[derive(Clone,Debug,Default,Eq,Hash,PartialEq)]
/// dimensions and strides for how a tensor data is layed out in memory
pub struct Layout{dims:Vec<usize>,stray:bool,strides:Vec<isize>}
#[cfg_attr(feature="serial",derive(Deserialize,Serialize))]
#[derive(Clone,Debug)]
/// tensor value type, similar to multidimensional Vec<T>. layout verification: tensor allows invalid layouts but behaves incorrectly or panics when used for anything, views have guaranteed layout validity
pub struct Tensor<E>{
	data:Vec<E>,							// buffer
	layout:Arc<Layout>,						// arrangement of components, reference counted to avoid having to clone for iterators
	#[cfg_attr(feature="serial",serde(skip))]
	views:Arc<ViewCache<E>>					// hack to allow indexing to get a &View even when one doesn't preexist to reference
}
unsafe impl<E:Send> Send for Tensor<E>{}			// implement send/sync based on E rather than the arena because that holds a cache of pointers rather than the actual data
unsafe impl<E:Sync> Sync for Tensor<E>{}
#[cfg(feature="serial")]
use serde::{Deserialize,Serialize};
use std::{
	borrow::{Borrow,BorrowMut},mem::{MaybeUninit,self},panic::RefUnwindSafe,ops::{Deref,DerefMut,Index,IndexMut,Range},sync::Arc
};
use super::{GridIter,Position,View,ViewCache,ViewIndex,index};
