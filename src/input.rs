use std::{error::Error, io::stdin};
use crate::agent::*;

pub fn assign(input: &mut [Neuron; INPS], inp: &[f64]) {
	for i in 0..INPS {
		input[i].excitation = inp[i] * crate::RESOLUTION // upscale by `RESOLUTION`
	}
}

pub fn inputs() -> Result<[f64; INPS*4], Box<dyn Error>> {
	let mut inp = [0.0; INPS*4];
	let mut csv = csv::Reader::from_reader(stdin());

	// Parse the CSV row-by-row and save as input
	for (i, res) in csv.deserialize().enumerate() {
		let rec: (f64, f64) = res?;

		inp[i] = rec.1 - rec.0
	}

	Ok(inp)
}
