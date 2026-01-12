/// find the lowest element >= than x
pub fn ceil<'a,E:PartialOrd<E>+PartialOrd<X>,X>(e:&'a View<E>,x:&X)->Option<&'a E>{// TODO we could return early if e==x
	let mut candidate=None;
	for ix in e.indices(){
		let e=&e[ix];
		if e>=x&&candidate.map(|c|c>e).unwrap_or(true){candidate=Some(e)}
	}

	candidate
}
/// find the lowest element >= than x
pub fn ceil_index<'a,E:PartialOrd<E>+PartialOrd<X>,X>(e:&'a View<E>,x:&X)->Option<(&'a E,Position)>{
	let mut candidate=None;
	for ix in e.indices(){
		let e=&e[&ix];
		if e>=x&&candidate.as_ref().map(|(c,_ix)|*c>e).unwrap_or(true){candidate=Some((e,ix))}
	}

	candidate
}
/// find the highest element <= x
pub fn floor<'a,E:PartialOrd<E>+PartialOrd<X>,X>(e:&'a View<E>,x:&X)->Option<&'a E>{
	let mut candidate=None;
	for ix in e.indices(){
		let e=&e[ix];
		if e<=x&&candidate.map(|c|c<e).unwrap_or(true){candidate=Some(e)}
	}

	candidate
}
/// find the highest element <= x
pub fn floor_index<'a,E:PartialOrd<E>+PartialOrd<X>,X>(e:&'a View<E>,x:&X)->Option<(&'a E,Position)>{
	let mut candidate=None;
	for ix in e.indices(){
		let e=&e[&ix];
		if e<=x&&candidate.as_ref().map(|(c,_ix)|*c<e).unwrap_or(true){candidate=Some((e,ix))}
	}

	candidate
}
/// excel style xmatch function TODO level 2 modes, faith checking, testing
pub fn xmatch<E:Display+PartialOrd<E>+PartialOrd<X>,X:Display>(query:&X,values:&View<E>,mmode:i32,smode:i32)->Option<Position>{
	if mmode.abs()==2||smode.abs()==2{todo!()}
	let (e,x)=(values,query);
	let mut candidate=None;

	for ix in (smode<0).then(||values.indices().rev()).into_iter().flat_map(|x|x).chain((smode>0).then(||values.indices()).into_iter().flat_map(|x|x)){
		let e=&e[&ix];

		match e.partial_cmp(x){
			None=>continue,
			Some(Ordering::Equal)=>return Some(ix),
			Some(Ordering::Greater)=>if mmode<0{continue},
			Some(Ordering::Less)=>if mmode>0{continue},
		}

		if candidate.as_ref().map(|(best,_index)|mmode>0&&*best>e||mmode<0&&*best<e).unwrap_or(true){candidate=Some((e,ix))}
	}
	return candidate.map(|(_e, ix)|ix);
}
use crate::builtin_tensor::{Position,View};
use std::{
	cmp::{Ordering,PartialOrd},fmt::Display
};
