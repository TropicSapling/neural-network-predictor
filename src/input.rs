use std::{error::Error, io::stdin};

use crate::agent::*;

pub fn assign(input: &mut [Neuron; INPS], j: usize) -> Result<(), Box<dyn Error>> {
	let mut csv = csv::Reader::from_reader(stdin());

	/*for (i, res) in csv.deserialize().enumerate() {
		let rec: (f64, f64) = res?;

		input[i].excitation = rec.1 - rec.0
	}*/

	for i in 0..INPS {
		input[i].excitation = (j as f64)%52.0; // placeholder
	}

	Ok(())
}
