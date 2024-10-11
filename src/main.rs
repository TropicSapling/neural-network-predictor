mod helpers;

mod ai;
mod agent;
mod input;
mod output;

use agent::*;
use ai::update_ai;

fn print_agent(agent: &Agent) {
	println!("\nNeural Network: {:#?}", agent.brain);
	println!("\nagent.err = {}\n", 1.0/agent.inverr)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let targets = output::targets();

	let mut agents: Vec<Agent> = vec![];
	let mut invsum = 0.0;
	for i in 0..55296 {
		// Remove worse-performing majority of agents once in a while
		if crate::helpers::rand_range(0..576) == 0 {
			// TODO: use minimax for this instead
			agents.sort_by(|a, b| b.inverr.partial_cmp(&a.inverr).unwrap());
			agents.truncate(48);
			invsum = 0.0;
			for agent in &mut agents {
				//agent.reset();
				update_ai(agent, targets[i], &mut invsum, i)?
			}

			let top = &agents[0];
			println!("top-err={}, gen={}, input={}, target={}", 1.0/top.inverr, top.brain.generation, i%52, targets[i])
		}

		agents.push(Agent::new(&agents, invsum));
		update_ai(agents.last_mut().unwrap(), targets[i], &mut invsum, i)?
	}

	// Print top agent
	agents.sort_by(|a, b| b.inverr.partial_cmp(&a.inverr).unwrap());
	agents.truncate(1);
	let mut errs = [0.0; 52];
	for i in 0..52 {
		update_ai(&mut agents[0], targets[i], &mut invsum, i)?;
		errs[i] = 1.0/agents[0].inverr
	}
	print_agent(&agents[0]);
	dbg!(errs);

	Ok(())
}
