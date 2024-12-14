mod ai;
mod data;
mod debug;
mod input;
mod output;
mod trainer;

mod agent;
mod consts;
mod helpers;

use std::io::Result;
use data::*;

/// TODO: PARALLELISM - bulk data processing
/// 
/// For every neuron *in parallel*, sequentially:
/// 1. Initiate *local* `neuron_excitations = vec![0; neurons.len()]`
/// 2. Check if neuron should be activated
/// 3. If so, for each connection: `neuron_excitations[recv_neuron] += conn.charge`
/// 
/// Then, once every neuron has finished, sequentially:
/// 1. Collect all `neuron_excitations` and sum them together into `total_excitations`
/// 2. Assign `total_excitations[neuron]` to each corresponding neuron *in parallel*

fn main() -> Result<()> {
	println!("");

	let mut agents = vec![];
	let input_data = read_data()?;
	
	// Train agents on the input data
	trainer::train(&mut agents, input_data, 53248);
	// note: currently there is practically no improvement beyond 53248*2=106496

	// Print final top agent
	debug::result(&mut agents[0], input_data)
}
