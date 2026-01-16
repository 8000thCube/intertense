impl<E:Default,S:AsRef<Path>> LoadSheet<S> for IOResult<Tensor<E>> where Tensor<E>:for<'a> ReadSheet<&'a Spreadsheet>{
	fn load_sheet(sheetpath:S)->Self{
		let mut result=Ok(Default::default());

		result.read_sheet(&mut [0,0,0],sheetpath.as_ref());
		result
	}
}
impl<E:Default,S:AsRef<Path>> LoadSheet<S> for Tensor<E> where Tensor<E>:for<'a> ReadSheet<&'a Path>{
	fn load_sheet(sheetpath:S)->Self{Self::from_sheet(sheetpath.as_ref())}
}
impl<E:Default> From<Spreadsheet> for Tensor<E> where Tensor<E>:ReadSheet<Spreadsheet>{
	fn from(sheet:Spreadsheet)->Tensor<E>{Self::from_sheet(sheet)}
}
impl<E:Default> From<Worksheet> for Tensor<E> where Tensor<E>:ReadSheet<Worksheet>{
	fn from(sheet:Worksheet)->Tensor<E>{Self::from_sheet(sheet)}
}
impl<E:Default> ReadSheet<&Path> for IOResult<Tensor<E>> where Tensor<E>:for<'a> ReadSheet<&'a Spreadsheet>{
	fn read_sheet(&mut self,indices:&mut [isize],sheet:&Path){
		let x=match xlsx::read(sheet){Err(e)=>return *self=Err(IOError::new(IOErrorKind::Other,e)),Ok(x)=>x};
		if let Ok(t)=self{return t.read_sheet(indices,&x)}

		let mut t:Tensor<E>=Default::default();
		t.read_sheet(indices,&x);

		*self=Ok(t)
	}
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
#[cfg(feature="match-tensor")]
impl CellPattern{
	/// builtin transformation cost function that uses levenstein*levscale for strings, xor*boolscale for bool, difference*diffscale for numbers, and parsescale when the string is not parsable
	pub fn diff_lev_transform_cost<S:AsRef<str>>(&self,boolscale:f32,diffscale:f32,levscale:f32,parsescale:f32,s:S)->f32{
		static METRIC:OnceLock<Levenshtein>=OnceLock::new();
		let s=s.as_ref();

		match self{
			CellPattern::Anything=>0.0,
			CellPattern::Bool(e)=>if let Ok(x)=s.parse::<bool>(){((e^x) as usize) as f32*boolscale}else{parsescale},
			CellPattern::Boolean=>if let Ok(_x)=s.parse::<bool>(){0.0}else{parsescale},
			CellPattern::Nothing=>if s.len()==0{0.0}else{parsescale},
			CellPattern::Number(e)=>if let Ok(x)=s.parse::<f64>(){((e-x).abs()*diffscale as f64) as f32}else{parsescale},
			CellPattern::Numeric=>if let Ok(_x)=s.parse::<f64>(){0.0}else{parsescale},
			CellPattern::Text(z)=>METRIC.get_or_init(Levenshtein::new).distance(s,z) as f32*levscale
		}
	}
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
		let (value, cost)=match_tensor::grab_table(&numbers,offsetcandidates,|a:&f32,b:&f32|{
			if a.is_nan()&&b.is_nan(){0.0}else if a.is_nan()||b.is_nan(){100.0}else{(a-b).abs()}
		},query.view()).unwrap();

		assert_eq!(value.swap_dims(-1,-2).flat_vec(None),expected);
		assert_eq!(cost,expected.into_iter().map(|e|(e-70.0).abs()).sum::<f32>());
	}
	#[test]
	fn find_table(){

		let expected=vec![
			"Single Investment Property",	"",						"",			"",						"",
			"Maximum LTV/CLTV",				"",						">= 1.00",	"",						"",
			"Minimum Credit Score",			"Maximum Loan Amount",	"Purchase",	"Rate/Term Refinance",	"Cash-Out Refinance",
			"700",							"1000000",				"80",		"80",					"80*",
			"",								"1500000",				"80",		"75",					"75",
			"",								"2000000",				"75",		"70",					"70",
			"",								"3000000",				"70",		"65",					"65",
			"",								"3500000",				"70",		"65",					"NA",
			"680",							"1000000",				"75",		"75",					"75",
			"",								"1500000",				"75",		"75",					"75",
			"",								"2000000",				"75",		"75",					"75",
			"",								"2500000",				"70",		"65",					"65",
			"",								"3000000",				"65",		"NA",					"NA",
			"640",							"1000000",				"75",		"70",					"70",
			"",								"1500000",				"70",		"70",					"70",
			"",								"2000000",				"65",		"NA",					"NA",
			"",								"3000000",				"60",		"NA",					"NA",
		];
		let matrixfile="Matrix November 10th FINAL.xlsx";
		let pattern=vec![
			CellPattern::Text("Single Investment Property".into()),CellPattern::Anything,CellPattern::Anything,CellPattern::Anything,CellPattern::Anything,
			CellPattern::Text("Maximum LTV".into()),CellPattern::Anything,CellPattern::Text(">=1.0".into()),CellPattern::Anything,CellPattern::Anything,
			CellPattern::Text("Minimum credit score".into()),CellPattern::Text("Maximum Loan Amount".into()),CellPattern::Text("Purchase".into()),CellPattern::Text("Rate/Term Refinance".into()),CellPattern::Text("Cash-Out Refinance".into()),
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
		];

		let mut data:Tensor<String>=Tensor::load_sheet(matrixfile);
		let mut pattern:Tensor<CellPattern>=Tensor::from(pattern);

		assert!(data.slice(&[0..3,0..5,0..30]));
		assert!(pattern.reshape([17,5]));
		assert!(pattern.swap_dims(0,1));

		let (table, _cost)=grab_table_default(data,pattern);

		assert_eq!(table.view().swap_dims(-1,-2).flat_vec(None),expected);
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
#[derive(Clone,Debug,Default,PartialEq)]
/// cell type usable for pattern matching
pub enum CellPattern{
	#[default]
	/// any value
	Anything,
	/// parable bool equal to
	Bool(bool),
	/// parsable bool
	Boolean,
	/// specifically an empty cell
	Nothing,
	/// parsable number equal to
	Number(f64),
	/// parsable number
	Numeric,
	/// text equal to
	Text(String)
}
#[cfg(feature="match-tensor")]
/// fuzzy finds a table based on the cell pattern
pub fn grab_table(data:impl AsRef<View<String>>,boolscale:f32,diffscale:f32,levscale:f32,parsescale:f32,pattern:impl AsRef<View<CellPattern>>)->(Tensor<String>,f32){
	let (data,mut pattern)=(data.as_ref(),pattern.as_ref());

	assert!(data.rank()>=pattern.rank());
	while data.rank()>pattern.rank(){pattern=pattern.unsqueeze_dim(0)}

	let (datadims,patterndims)=(data.dims(),pattern.dims());
	let mut stopposition=vec![0;datadims.len()];

	datadims.iter().zip(patterndims.iter()).zip(stopposition.iter_mut()).for_each(|((d,p),s)|*s=d.checked_sub(*p).unwrap()+1);
	let offsetcandidates=GridIter::new(stopposition);

	match_tensor::grab_table(data,offsetcandidates,|d,p|p.diff_lev_transform_cost(boolscale,diffscale,levscale,parsescale,d),pattern).map(|(t, c)|(t.to_tensor(), c)).unwrap()
}
#[cfg(feature="match-tensor")]
/// fuzzy finds a table based on the cell pattern
pub fn grab_table_default(data:impl AsRef<View<String>>,pattern:impl AsRef<View<CellPattern>>)->(Tensor<String>,f32){grab_table(data,1.0,1.0,1.0,5.0,pattern)}
/// excel style xmatch function TODO level 2 modes, faith checking, testing
pub fn xmatch<E:Display+PartialOrd<E>+PartialOrd<X>,X:Display>(query:&X,values:&View<E>,mmode:i32,smode:i32)->Option<Position>{xmatch::xmatch(query,values,mmode,smode)}
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
#[cfg(feature="match-tensor")]
use b_k_tree::{metrics::Levenshtein,DiscreteMetric};
#[cfg(feature="match-tensor")]
use crate::builtin_tensor::GridIter;
#[cfg(feature="match-tensor")]
use crate::match_tensor;
use crate::{
	builtin_tensor::{Position,Tensor,View},xmatch
};
use std::{
	fmt::Display,io::{Error as IOError,ErrorKind as IOErrorKind,Result as IOResult},path::Path
};
#[cfg(feature="match-tensor")]
use std::sync::OnceLock;
use umya_spreadsheet::{Cell,Spreadsheet,Worksheet,reader::xlsx};
