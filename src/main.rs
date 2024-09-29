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
	let mut highscore = 0.0;
	for _ in 0..256 {
		update_ai(&mut agents, 456.0 - 123.0 /* placeholder */, &mut highscore)
	}

	//print_agent(&agents, agents.last().unwrap())
}
