#[cfg(test)]
mod tests{
	#[test]
	fn absorb_00(){
		let offsetcandidates=GridIter::new([2,3]);
		let mut data:Tensor<f32>=vec![
			0.5,0.0,0.5,0.4,
			0.4,0.1,0.4,0.4,
			0.3,0.2,0.4,0.5,
		].into();
		let mut query:Tensor<f32>=vec![	// approximately this appears at 1,1
			0.2,0.4,
			0.2,0.4,
		].into();

		assert!(data.reshape([3,4])&&query.reshape([2,2]));
		let (value, cost)=absorb_data(data.view(),|x0,x1|x0.iter().zip(x1.iter()).map(|(x0,x1)|(x0-x1) as f32).map(|x|x*x).sum::<f32>().sqrt(),offsetcandidates,|a,b|(a-b).abs(),query.view()).unwrap();

		assert_eq!((cost,value.flat_vec(None)),(0.1,vec![0.1,0.4,0.2,0.4]));
	}

	use super::*;
}
/// find the lowest cost offset from the candidates and extract the corresponding components from the data. returns none if length of offsetcandidates is 0. having any offset candidates that put any of query outside of data is currently not supported. data will have to be padded first if that is desired
pub fn absorb_data<E,F:FnMut(&Position,&Position)->f32,I:IntoIterator<Item=Position>,T:FnMut(&E,&X)->f32,X>(data:&View<E>,mut movecost:F,offsetcandidates:I,mut transformcost:T,query:&View<X>)->Option<(Tensor<E>,f32)> where E:Clone{
	let mut qx=Position::new(0);
	let mut sx=Vec::new();

	let offsetcandidates=offsetcandidates.into_iter();
	let mut result:Option<(Tensor<(Position,f32)>,f32)>=None;

	for offset in offsetcandidates{
		let dims=query.dims();

		sx.clear();
		sx.extend((0..query.rank()).map(|n|{
			let d=dims[n];
			let o:usize=offset[n].try_into().unwrap();

			o as isize..(d+o) as isize
		}));

		let mut acc=data.map_indexed(|e,ix|{
			qx.clone_from(&ix);
			qx-=&offset;

			let cost=query.get(qx.as_unsigned()).map(|q|transformcost(e,q)).unwrap_or(f32::INFINITY);
			(ix, cost)
		});
		propagate_cost(&mut acc,|ix,jx|{
			qx.clone_from(&ix);
			qx-=&offset;

			query.get(qx.as_unsigned()).map(|q|movecost(&jx,&ix)+transformcost(&data[jx],q)).unwrap_or(f32::INFINITY)
		});

		assert!(acc.slice(&sx),"{sx:?}");

		let cost:f32=acc.indices().map(|ix|acc[ix].1).sum();
		if result.as_ref().map(|(_a,c)|cost<*c).unwrap_or(true){result=Some((acc,cost))}
	}
	result.map(|(a,c)|(a.map(|(position,_cost)|data[position].clone()),c))
}
/// find the lowest element greater than x
pub fn ceil<'a,E:PartialOrd<E>+PartialOrd<X>,X>(e:&'a View<E>,x:&X)->Option<&'a E>{
	let mut candidate=None;
	for ix in e.indices(){
		let e=&e[ix];
		if e>x&&candidate.map(|c|c>e).unwrap_or(true){candidate=Some(e)}
	}

	candidate
}
/// find the lowest element greater than x
pub fn ceil_index<'a,E:PartialOrd<E>+PartialOrd<X>,X>(e:&'a View<E>,x:&X)->Option<(&'a E,Position)>{
	let mut candidate=None;
	for ix in e.indices(){
		let e=&e[&ix];
		if e>x&&candidate.as_ref().map(|(c,_ix)|*c>e).unwrap_or(true){candidate=Some((e,ix))}
	}

	candidate
}
/// fills holes in the data using data from the left. holes with no left non holes will remain
pub fn fill_holes<E,F:FnMut(&E)->E,G:FnMut(&E)->bool>(data:&mut View<E>,dim:isize,mut fill_hole:F,mut is_hole:G){
	let data=data.swap_dims_mut(dim,0);
	let mut left=Position::new(0);

	for ix in data.indices(){
		if is_hole(&data[&ix]){
			if left.len()>0&&left[1..]==ix[1..]{data[&ix]=fill_hole(&data[&left])}
		}else{
			left.clone_from(&ix);
		}
	}
}
/// find the highest element less than x
pub fn floor<'a,E:PartialOrd<E>+PartialOrd<X>,X>(e:&'a View<E>,x:&X)->Option<&'a E>{
	let mut candidate=None;
	for ix in e.indices(){
		let e=&e[ix];
		if e<x&&candidate.map(|c|c<e).unwrap_or(true){candidate=Some(e)}
	}

	candidate
}
/// find the highest element less than x
pub fn floor_index<'a,E:PartialOrd<E>+PartialOrd<X>,X>(e:&'a View<E>,x:&X)->Option<(&'a E,Position)>{
	let mut candidate=None;
	for ix in e.indices(){
		let e=&e[&ix];
		if e<x&&candidate.as_ref().map(|(c,_ix)|*c<e).unwrap_or(true){candidate=Some((e,ix))}
	}

	candidate
}
/// euclidean distance move cost TODO make this trait based so it works for tensors
pub fn euclidean(p:&Position,q:&Position)->f32{p.iter().zip(q.iter()).map(|(p,q)|(p-q) as f32).map(|x|x*x).sum::<f32>().sqrt()}
/// fuzzy equality based on which components match according to a predicate. broadcasting and rank promotion are performed, but not padding. dim mismatch after broadcasting and rank promotion will cause a result of false
pub fn match_eq<E,F:FnMut(&E,&X)->bool,X>(mut e:&View<E>,mut f:F,mut x:&View<X>)->bool{
	if e.dims().iter().rev().zip(x.dims().iter().rev()).any(|(&d,&x)|d!=x&&d!=1&&x!=1){return false}

	while e.rank()<x.rank(){e=e.unsqueeze_dim(0)}
	while e.rank()>x.rank(){x=x.unsqueeze_dim(0)}

	let (edims,xdims)=(e.dims(),x.dims());
	for n in 0..edims.len(){
		if edims[n]==1{e=e.broadcast_dim(n as isize,xdims[n])}
		if xdims[n]==1{x=x.broadcast_dim(n as isize,edims[n])}
	}

	e.indices().all(|ix|f(&e[&ix],&x[&ix]))
}
/// finds which indices match according to the predicate. broadcasting, padding, and rank promotion are performed // TODO those could be optional
pub fn matches<'a,E,F:'a+FnMut(&E,&X)->bool,X>(mut e:&'a View<E>,mut f:F,mut x:&'a View<X>)->FilterMap<GridIter,impl 'a+FnMut(Position)->Option<Position>>{
	while e.rank()<x.rank(){e=e.unsqueeze_dim(0)}
	while e.rank()>x.rank(){x=x.unsqueeze_dim(0)}

	let (edims,xdims)=(e.dims(),x.dims());
	for n in 0..edims.len(){
		if edims[n]==1{e=e.broadcast_dim(n as isize,xdims[n])}
		if xdims[n]==1{x=x.broadcast_dim(n as isize,edims[n])}
	}

	let mut dims=edims.to_vec();
	for n in 0..edims.len(){dims[n]=dims[n].max(xdims[n])}

	GridIter::new(dims).filter_map(move|ix|f(e.get(&ix)?,x.get(&ix)?).then_some(ix))
}
/// propagate movement costs // TODO wrap args?
pub fn propagate_cost<F:FnMut(&Position,&Position)->f32>(acc:&mut View<(Position,f32)>,mut cost:F){
	let mut adjacent=Position::new(acc.rank());

	for ix in acc.indices().chain(acc.indices().rev()){
		adjacent.copy_from_slice(&ix);
		for n in 0..adjacent.len(){
			let dn=acc.dims()[n] as isize;

			adjacent[n]=ix[n]-1;
			let (jx,c)=if let Some((jx,_c))=(adjacent[n]>=0).then(||&acc[&adjacent]){(jx.clone(), cost(jx, &ix))}else{(ix.clone(), f32::INFINITY)};

			adjacent[n]=ix[n]+1;
			let (kx,d)=if let Some((kx,_c))=(adjacent[n]<dn).then(||&acc[&adjacent]){(kx.clone(), cost(kx, &ix))}else{(ix.clone(), f32::INFINITY)};

			adjacent[n]=ix[n];
			let (ix,b)=&mut acc[&ix];

			if *b>c{(*ix,*b)=(jx,c)}
			if *b>d{(*ix,*b)=(kx,d)}
		}
	}
}
use crate::builtin_tensor::{GridIter,Position,Tensor,View};
use std::{cmp::PartialOrd,iter::FilterMap};
