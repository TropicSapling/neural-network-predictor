use std::{error::Error, io::stdin};

pub const DATA_SIZE: usize = crate::INPS*(crate::PARTITIONS+5);

pub fn read_data() -> Result<[f64; DATA_SIZE], Box<dyn Error>> {
	let mut inp = [0.0; DATA_SIZE];
	let mut csv = csv::Reader::from_reader(stdin());

	// Parse the CSV row-by-row and save as i/o data
	let mut prev = 0.0;
	for (i, res) in csv.deserialize().enumerate() {
		let rec: (f64, f64) = res?;

		if i > DATA_SIZE {
			break
		} else if i > 0 {
			inp[DATA_SIZE - i] = prev - rec.1
		}

		prev = rec.0
	}

	Ok(inp)
}
