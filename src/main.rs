mod helpers;

mod ai;
mod agent;
mod input;
mod output;

use agent::*;
use ai::update_ai;

fn print_agent(agents: &Vec<Agent>, agent: &Agent) {
	println!("\nNeural Network: {:#?}", agent.brain);
	println!("\nagent.err = {}\n\nAGENTS: {}", 1.0/agent.inverr, agents.len())
}

fn main() {
	let mut agents: Vec<Agent> = vec![];
	let mut invsum = 0.0;
	let mut hs     = 0.0;
	for i in 0..13824 {
		if invsum == f64::INFINITY {break}

		// Remove worse-performing majority of agents once in a while
		if i % 576 == 575 {
			agents.sort_by(|a, b| b.inverr.partial_cmp(&a.inverr).unwrap());
			agents.truncate(24);
			println!("mid-err={}", 1.0/agents.last().unwrap().inverr)
		}

		agents.push(Agent::new(&agents, invsum));
		update_ai(agents.last_mut().unwrap(), 3.1415926535, &mut invsum, &mut hs)
	}

	// Print top agent
	agents.sort_by(|a, b| b.inverr.partial_cmp(&a.inverr).unwrap());
	agents.truncate(1);
	print_agent(&agents, agents.last().unwrap())
}
