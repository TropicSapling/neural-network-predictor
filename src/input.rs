use std::{error::Error, io::stdin};

use crate::agent::*;

pub fn assign(input: &mut [Neuron; INPS]) -> Result<(), Box<dyn Error>> {
	let mut csv = csv::Reader::from_reader(stdin());

	for (i, res) in csv.deserialize().enumerate() {
		let rec: (f64, f64) = res?;

		input[i].excitation = rec.1 - rec.0
	}

	Ok(())
}
