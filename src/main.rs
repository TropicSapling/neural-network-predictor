mod helpers;

mod ai;
mod agent;
mod input;
mod output;

use agent::*;
use ai::update_ai;

fn print_agent(agent: &mut Agent, inputs: [f64; INPS], targets: [f64; INPS]) {
	println!("\nNeural Network: {:#?}\n\nmaxerr = {}\n", agent.brain, agent.maxerr);

	for i in 0..INPS {
		let predictions = update_ai(agent.reset(), inputs[i], targets[i]);
		let err         = (predictions[1] - predictions[0] - targets[i]).abs();

		println!("{predictions:?} => {} (err={err})", predictions[1] - predictions[0])
	}
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let inputs  = input::inputs()?;
	let targets = output::targets();

	let mut agents: Vec<Agent> = vec![];
	let mut invsum = 0.0;
	let mut prverr = 0.0;
	let mut stayed = 0;
	for n in 0..65536 {
		agents.push(Agent::from(&agents, invsum, invsum)); // TODO: minsum & maxsum
		for i in 0..INPS {
			update_ai(agents.last_mut().unwrap().reset(), inputs[i], targets[i]);
		}

		invsum += 1.0/agents.last().unwrap().maxerr;

		// Remove worse-performing majority of agents once in a while
		if n % 256 == 0 {
			agents.sort_by(|a, b| a.maxerr.partial_cmp(&b.maxerr).unwrap());
			agents.truncate(16);

			invsum = 0.0;
			for agent in &agents {
				invsum += 1.0/agent.maxerr
			}

			let top = &agents[0];
			println!("maxerr={}, gen={}", top.maxerr, top.brain.generation);

			// Quit training if things have started to converge
			if top.maxerr == prverr {
				stayed += 1;
				if stayed > 31 || top.maxerr == 0.0 {
					println!("\nn={n}");
					break
				}
			} else {
				prverr = top.maxerr;
				stayed = 0
			}
		}
	}

	// Print top agent
	agents.sort_by(|a, b| a.maxerr.partial_cmp(&b.maxerr).unwrap());
	agents.truncate(1);
	print_agent(&mut agents[0], inputs, targets);

	Ok(())
}
