mod data;
mod debug;
mod helpers;

mod ai;
mod agent;
mod input;
mod output;
mod trainer;

use std::error::Error;
use agent::*;
use data::*;

// All I/O is upscaled/downscaled by 128x
const RESOLUTION: f64 = 128.0;
// Partitions for cross-validation
const PARTITIONS: usize = 2;

fn main() -> Result<(), Box<dyn Error>> {
	println!("");

	let mut agents = vec![];
	let input_data = read_data()?;
	
	// Train agents on the input data
	trainer::train(&mut agents, input_data, 53248);

	// Print final top agent
	debug::result(&mut agents[0], input_data);

	Ok(())
}
