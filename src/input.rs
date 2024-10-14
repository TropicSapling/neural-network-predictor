use std::{error::Error, io::stdin};
use crate::agent::*;

pub fn assign(input: &mut [Neuron; INPS], inp: f64) {
	for i in 0..INPS {
		input[i].excitation = inp // for now; should be different per neuron
	}
}

pub fn inputs() -> Result<[f64; INPS*2], Box<dyn Error>> {
	let mut inp = [0.0; INPS*2];
	let mut csv = csv::Reader::from_reader(stdin());

	for (i, res) in csv.deserialize().enumerate() {
		let rec: (f64, f64) = res?;
		if i >= INPS*2 {
			break
		}

		inp[i] = rec.1 - rec.0
	}

	Ok(inp)
}
