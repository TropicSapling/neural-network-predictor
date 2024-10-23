use std::{error::Error, io::stdin};
use crate::agent::*;

pub fn assign(input: &mut [Neuron; INPS], inp: &[f64]) {
	for i in 0..INPS {
		input[i].excitation = inp[i] * crate::RESOLUTION // upscale by `RESOLUTION`
	}
}

pub fn inputs() -> Result<[f64; INPS*6], Box<dyn Error>> {
	let mut inp = [0.0; INPS*6];
	let mut csv = csv::Reader::from_reader(stdin());

	// Parse the CSV row-by-row and save as input
	let mut prev = 0.0;
	for (i, res) in csv.deserialize().enumerate() {
		let rec: (f64, f64) = res?;

		if i > INPS*6 {
			break
		} else if i > 0 {
			inp[INPS*6 - i] = prev - rec.1
		}

		prev = rec.0
	}

	Ok(inp)
}
