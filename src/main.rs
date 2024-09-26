mod structs;
mod helpers;
mod ai;

use structs::*;
use ai::update_ai;

pub fn print_agent(agents: Vec<Agent>, agent: Agent) {
	println!("Neural Network: {:#?}\n\nAGENTS ALIVE: {}", agent.brain, agents.len())
}

fn main() {
	let mut agents = vec![];
	for _ in 0..256 {
		agents.push(Agent::new(&agents));
		update_ai(&mut agents)
	}
}
