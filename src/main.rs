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

// TODO - fix cross-validation:
// - Stop sorting. Just get best train error agent.
// - Save best train agent's validation error every epoch
// - Check its validation error
// - IF WORSE THAN SAVED: Discard all new agents and switch training set
// - IF OK: Restore train error & then optimise

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
