use std::{error::Error, io::stdin};
use crate::consts::*;

pub type DataRow = [f64; OUTS];
pub type Data    = [DataRow; DATA_SIZE];

pub fn read_data() -> Result<Data, Box<dyn Error>> {
	let mut inp = [[0.0; OUTS]; DATA_SIZE];
	let mut csv = csv::Reader::from_reader(stdin());

	// Parse the CSV row-by-row and save as i/o data
	for (i, res) in csv.deserialize().enumerate() {
		if i >= DATA_SIZE {
			break
		}

		inp[i] = res?
	}

	Ok(inp)
}
