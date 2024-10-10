mod helpers;

mod ai;
mod agent;
mod input;
mod output;

use agent::*;
use ai::update_ai;

fn print_agent(agent: &Agent) {
	println!("\nNeural Network: {:#?}", agent.brain);
	println!("\nagent.err = {}", 1.0/agent.inverr)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let targets = output::targets();

	let mut agents: Vec<Agent> = vec![];
	let mut invsum = 0.0;
	let mut hs     = 0.0;
	for i in 0..13824 {
		if invsum == f64::INFINITY {break}

		// Remove worse-performing majority of agents once in a while
		if i % 576 == 575 {
			agents.sort_by(|a, b| b.inverr.partial_cmp(&a.inverr).unwrap());
			agents.truncate(24);
			invsum = 0.0;
			for agent in &mut agents {
				//agent.reset();
				update_ai(agent, targets[i], &mut invsum, &mut hs)?
			}

			let mid = agents.last().unwrap();
			println!("mid-err={}, gen={}, invsum={invsum}", 1.0/mid.inverr, mid.brain.generation)
		}

		agents.push(Agent::new(&agents, invsum));
		update_ai(agents.last_mut().unwrap(), targets[i], &mut invsum, &mut hs)?
	}

	// Print top agent
	agents.sort_by(|a, b| b.inverr.partial_cmp(&a.inverr).unwrap());
	agents.truncate(24);
	invsum = 0.0;
	for agent in &mut agents {
		//agent.reset();
		update_ai(agent, targets[13822], &mut invsum, &mut hs)?
	}
	print_agent(agents.last().unwrap());

	Ok(())
}
