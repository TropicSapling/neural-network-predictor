mod ai;
mod data;
mod debug;
mod input;
mod output;
mod trainer;

mod agent;
mod consts;
mod helpers;

use std::error::Error;
use data::*;

fn main() -> Result<(), Box<dyn Error>> {
	println!("");

	let mut agents = vec![];
	let input_data = read_data()?;
	
	// Train agents on the input data
	trainer::train(&mut agents, input_data, 26624);

	// Print final top agent
	debug::result(&mut agents[0], input_data);

	Ok(())
}
