impl<E:Default,S:AsRef<Path>> LoadSheet<S> for Tensor<E> where Tensor<E>:for<'a> ReadSheet<&'a Path>{
	fn load_sheet(sheetpath:S)->Self{Self::from_sheet(sheetpath.as_ref())}
}
impl<E:Default> From<Spreadsheet> for Tensor<E> where Tensor<E>:ReadSheet<Spreadsheet>{
	fn from(sheet:Spreadsheet)->Tensor<E>{Self::from_sheet(sheet)}
}
impl<E:Default> From<Worksheet> for Tensor<E> where Tensor<E>:ReadSheet<Worksheet>{
	fn from(sheet:Worksheet)->Tensor<E>{Self::from_sheet(sheet)}
}
impl<E:Default> ReadSheet<&Path> for Tensor<E> where Tensor<E>:for<'a> ReadSheet<&'a Spreadsheet>{
	fn read_sheet(&mut self,indices:&mut [isize],sheet:&Path){
		let x=xlsx::read(sheet).unwrap();
		self.read_sheet(indices,&x);
	}
}
impl<E:Default> ReadSheet<&Spreadsheet> for Tensor<E> where Tensor<E>:for<'a> ReadSheet<&'a Worksheet>{
	fn read_sheet(&mut self,indices:&mut [isize],sheet:&Spreadsheet){
		assert!(indices.len()>=3);
		let im3=indices.len()-3;

		for n in 0..{
			indices[im3]=n as isize;
			self.read_sheet(indices,if let Some(s)=sheet.get_sheet(&n){s}else{break});
		}
	}
}
impl<E:Default> ReadSheet<Spreadsheet> for Tensor<E> where Tensor<E>:for<'a> ReadSheet<&'a Spreadsheet>{
	fn read_sheet(&mut self,indices:&mut [isize],sheet:Spreadsheet){self.read_sheet(indices,&sheet)}
}
impl<E:Default> ReadSheet<Worksheet> for Tensor<E> where Tensor<E>:for<'a> ReadSheet<&'a Worksheet>{
	fn read_sheet(&mut self,indices:&mut [isize],sheet:Worksheet){self.read_sheet(indices,&sheet)}
}
impl ReadSheet<&Worksheet> for Tensor<Option<Cell>>{
	fn read_sheet(&mut self,indices:&mut [isize],sheet:&Worksheet){
		assert!(indices.len()>=2);

		let (xstop,ystop)=sheet.get_highest_column_and_row();
		let im2=indices.len()-2;

		while indices.len()>self.rank(){assert!(self.unsqueeze_dim(0))}

		assert!(self.pad_dim(-2,xstop as usize,None)&&self.pad_dim(-1,ystop as usize,None));
		if im2>0{assert!(self.pad_dim(-3,(indices[im2-1]+1) as usize,None))}

		for x in 0..xstop{
			indices[im2]=x as isize;
			for y in 0..ystop{
				indices[im2+1]=y as isize;
				self[&indices[..]]=sheet.get_cell((x+1,y+1)).cloned();
			}
		}
	}
}
impl ReadSheet<&Worksheet> for Tensor<String>{
	fn read_sheet(&mut self,indices:&mut [isize],sheet:&Worksheet){
		assert!(indices.len()>=2);

		let (xstop,ystop)=sheet.get_highest_column_and_row();
		let im2=indices.len()-2;

		while indices.len()>self.rank(){assert!(self.unsqueeze_dim(0))}

		assert!(self.pad_dim(-2,xstop as usize,String::new())&&self.pad_dim(-1,ystop as usize,String::new()));
		if im2>0{assert!(self.pad_dim(-3,(indices[im2-1]+1) as usize,String::new()))}

		for x in 0..xstop{
			indices[im2]=x as isize;
			for y in 0..ystop{
				indices[im2+1]=y as isize;
				self[&indices[..]]=sheet.get_value((x+1,y+1));
			}
		}
	}
}
impl ReadSheet<&Worksheet> for Tensor<f32>{
	fn read_sheet(&mut self,indices:&mut [isize],sheet:&Worksheet){
		assert!(indices.len()>=2);

		let (xstop,ystop)=sheet.get_highest_column_and_row();
		let im2=indices.len()-2;

		while indices.len()>self.rank(){assert!(self.unsqueeze_dim(0))}

		assert!(self.pad_dim(-2,xstop as usize,f32::NAN)&&self.pad_dim(-1,ystop as usize,f32::NAN));
		if im2>0{assert!(self.pad_dim(-3,(indices[im2-1]+1) as usize,f32::NAN))}

		for x in 0..xstop{
			indices[im2]=x as isize;
			for y in 0..ystop{
				indices[im2+1]=y as isize;
				self[&indices[..]]=sheet.get_value_number((x+1,y+1)).unwrap_or(f64::NAN) as f32;
			}
		}
	}
}
impl ReadSheet<&Worksheet> for Tensor<f64>{
	fn read_sheet(&mut self,indices:&mut [isize],sheet:&Worksheet){
		assert!(indices.len()>=2);

		let (xstop,ystop)=sheet.get_highest_column_and_row();
		let im2=indices.len()-2;

		while indices.len()>self.rank(){assert!(self.unsqueeze_dim(0))}

		assert!(self.pad_dim(-2,xstop as usize,f64::NAN)&&self.pad_dim(-1,ystop as usize,f64::NAN));
		if im2>0{assert!(self.pad_dim(-3,(indices[im2-1]+1) as usize,f64::NAN))}

		for x in 0..xstop{
			indices[im2]=x as isize;
			for y in 0..ystop{
				indices[im2+1]=y as isize;
				self[&indices[..]]=sheet.get_value_number((x+1,y+1)).unwrap_or(f64::NAN);
			}
		}
	}
}

#[cfg(test)]
mod tests{
	#[cfg(feature="match-tensor")]
	#[test]
	fn match_matrix(){
		use crate::match_tensor;

		let expected=vec![80.0,80.0,80.0,75.0,75.0,70.0,70.0,65.0,70.0,65.0];
		let matrixfile="Matrix November 10th FINAL.xlsx";
		let numbers:Tensor<f32>=Tensor::load_sheet(matrixfile);

		assert!(numbers[[0,0,0]].is_nan());
		assert!(numbers[[0,0,1]].is_nan());
		assert!(numbers[[0,1,0]].is_nan());

		let mut query:Tensor<f32>=vec![70.0;expected.len()].into();

		assert!(query.reshape([1,2,5]));

		let offsetcandidates=numbers.indices().filter(|ix|ix[0]==1&&ix[1]<4&&ix[2]<4);
		let (value, cost)=match_tensor::absorb_data(&numbers,|x0,x1|x0.iter().zip(x1.iter()).map(|(x0,x1)|(x0-x1) as f32).map(|x|x*x).sum::<f32>().sqrt(),offsetcandidates,|a:&f32,b:&f32|{
			if a.is_nan()&&b.is_nan(){0.0}else if a.is_nan()||b.is_nan(){100.0}else{(a-b).abs()}
		},query.view()).unwrap();

		assert_eq!(value.view().swap_dims(-1,-2).flat_vec(None),expected);
		assert_eq!(cost,expected.into_iter().map(|e|(e-70.0).abs()).sum::<f32>());
	}
	#[test]
	fn read_matrix(){
		let matrixfile="Matrix November 10th FINAL.xlsx";
		let numbers:Tensor<f32>=Tensor::load_sheet(matrixfile);
		let text:Tensor<String>=Tensor::load_sheet(matrixfile);

		assert!(numbers[[0,0,0]].is_nan());
		assert!(numbers[[0,0,1]].is_nan());
		assert!(numbers[[0,1,0]].is_nan());
		assert_eq!(numbers[[1..2,2..4,3..8].as_slice()].swap_dims(-1,-2).flat_vec(None),vec![80.0,80.0,80.0,75.0,75.0,70.0,70.0,65.0,70.0,65.0]);
		assert_eq!(text[[0,0,1]],"test");
	}
	use super::*;
}

/// provide function for loading spreadsheet to tensor by path refs
pub trait LoadSheet<S>{
	/// loads a tensor from the spreadsheet file
	fn load_sheet(sheetpath:S)->Self;
}
/// how to read the columns of a spreadsheet into a tensor
pub trait ReadSheet<S>{
	/// reads the excel sheet into a tensor
	fn from_sheet(sheet:S)->Self where Self:Default{
		let mut tensor=Self::default();

		tensor.read_sheet(&mut [0,0,0],sheet);
		tensor
	}
	/// reads the tensor from excel format
	fn read_sheet(&mut self,indices:&mut [isize],sheet:S);
}
use crate::builtin_tensor::Tensor;
use std::path::Path;
use umya_spreadsheet::{Cell,Spreadsheet,Worksheet,reader::xlsx};
