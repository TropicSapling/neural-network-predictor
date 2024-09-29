mod structs;
mod helpers;
mod ai;

use structs::*;
use ai::update_ai;

fn print_agent(agents: &Vec<Agent>, agent: &Agent) {
	println!("Neural Network: {:#?}\n\nAGENTS ALIVE: {}", agent.brain, agents.len())
}

fn main() {
	let mut agents = vec![];
	for _ in 0..256 {
		update_ai(&mut agents, 4.56 - 1.23 /* placeholder */)
	}

	print_agent(&agents, agents.last().unwrap())
}
