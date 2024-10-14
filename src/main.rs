mod helpers;

mod ai;
mod agent;
mod input;
mod output;

use agent::*;
use ai::update_ai;

fn print_agent(agent: &mut Agent, inputs: [f64; INPS], targets: [f64; INPS]) {
	let (brain, maxerr, toterr) = (&agent.brain, agent.maxerr, agent.toterr);

	println!("\nNeural Network: {brain:#?}\n\nmaxerr = {maxerr}\ntoterr = {toterr}\n");

	agent.toterr = 0.0;
	agent.maxerr = 0.0;
	for i in 0..INPS {
		let out = update_ai(agent.reset(), inputs[i], targets[i]);
		let err = (out[1] - out[0] - targets[i]).abs();

		let inp = inputs[i];
		let tgt = targets[i];
		let res = out[1] - out[0];

		println!("{inp:<2} => {out:>5.1?} => {res:<5.1} (err={err:<5.1}) <= {tgt}")
	}
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let inputs  = input::inputs()?;
	let targets = output::targets();

	let mut agents: Vec<Agent> = vec![];
	let mut totsum = 0.0;
	let mut maxsum = 0.0;
	let mut prverr = 0.0;
	let mut stayed = 0;
	for n in 0..65536 {
		agents.push(Agent::from(&agents, totsum, maxsum));
		agents.last_mut().unwrap().toterr = 0.0;
		agents.last_mut().unwrap().maxerr = 0.0;
		for i in 0..INPS {
			update_ai(agents.last_mut().unwrap().reset(), inputs[i], targets[i]);
		}

		totsum += 1.0/agents.last().unwrap().toterr;
		maxsum += 1.0/agents.last().unwrap().maxerr;

		// Remove worse-performing majority of agents once in a while
		if n % 384 == 0 {
			agents.sort_by(|a, b| a.toterr.partial_cmp(&b.toterr).unwrap());
			agents.truncate(256);

			totsum = 0.0;
			maxsum = 0.0;
			for agent in &agents {
				totsum += 1.0/agent.toterr;
				maxsum += 1.0/agent.maxerr;
			}

			let top = &agents[0];
			println!("toterr={}, gen={}", top.toterr, top.brain.generation);

			// Quit training if things have started to converge
			if top.toterr == prverr {
				stayed += 1;
				if stayed > 23 || top.toterr == 0.0 {
					println!("\nn={n}");
					break
				}
			} else {
				prverr = top.toterr;
				stayed = 0
			}
		}
	}

	// Print top agent
	agents.sort_by(|a, b| a.toterr.partial_cmp(&b.toterr).unwrap());
	agents.truncate(1);
	print_agent(&mut agents[0], inputs, targets);

	Ok(())
}
