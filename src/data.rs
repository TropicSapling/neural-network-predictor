use std::{error::Error, io::stdin};

pub const DATA_SIZE: usize = crate::INPS*(crate::PARTITIONS+4);

pub fn read_data() -> Result<[f64; DATA_SIZE], Box<dyn Error>> {
	let mut inp = [0.0; DATA_SIZE];
	let mut csv = csv::Reader::from_reader(stdin());

	// Parse the CSV row-by-row and save as i/o data
	let mut i = 0;
	for res in csv.deserialize() {
		let rec: (f64, f64) = res?;
		if i >= DATA_SIZE {
			break
		}

		inp[DATA_SIZE - i - 1] = rec.0;
		inp[DATA_SIZE - i - 2] = rec.1;

		i += 2
	}

	Ok(inp)
}
