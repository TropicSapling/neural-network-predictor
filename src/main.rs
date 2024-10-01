mod structs;
mod helpers;
mod ai;

use structs::*;
use ai::update_ai;

fn print_agent(agents: &Vec<Agent>, agent: &Agent) {
	println!("\nNeural Network: {:#?}", agent.brain);
	println!("\nagent.err = {}\n\nAGENTS: {}", 1.0/agent.inverr, agents.len())
}

fn main() {
	let mut agents = vec![];
	let mut invsum = 0.0;
	let mut hs     = 0.0;
	for _ in 0..8192 {
		if invsum == f64::INFINITY {break}

		agents.push(Agent::new(&agents, invsum));
		update_ai(agents.last_mut().unwrap(), 456.0 - 123.0, &mut invsum, &mut hs)
		// TODO: remove agent with worst error 1/3 of the time
	}

	print_agent(&agents, agents.last().unwrap())
}
