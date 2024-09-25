#[macro_use]
mod structs;
mod helpers;
mod ai;

use {helpers::*, structs::*};
use ai::update_ai;

const INV_SPAWN_RATE: usize = 32;

pub fn print_agent(agents: Vec<Agent>, agent: Agent) {
	println!("Neural Network: {:#?}\n\nAGENTS ALIVE: {}", agent.brain, agents.len())
}

fn main() {
	let mut agents = vec![];

	// Randomly spawn new agents
	if rand_range(0..INV_SPAWN_RATE) == 0 {
		let agent = Agent::new(&agents);
		agents.push(agent)
	}

	if let Some(agent) = Agent::maybe_split(&mut agents) {
		agents.push(agent)
	}

	update_ai(&mut agents)
}
