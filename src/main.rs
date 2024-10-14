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

		println!("{inp:<2} => {out:>6.1?} => {res:<6.2} (err={err:<5.2}) <= {tgt:.2}")
	}
}

fn optimise(agents: &mut Vec<Agent>) {
	let rank = |agent: &Agent| agent.maxerr.powf(2.0) + agent.toterr;

	agents.sort_by(|a, b| rank(a).partial_cmp(&rank(b)).unwrap());
	agents.truncate(192);
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

		let agent = agents.last_mut().unwrap();

		agent.toterr = 0.0;
		agent.maxerr = 0.0;
		for i in 0..INPS {
			update_ai(agent.reset(), inputs[i], targets[i]);
		}

		totsum += 1.0/agent.toterr;
		maxsum += 1.0/agent.maxerr;

		// Remove worse-performing majority of agents once in a while
		if n % 384 == 0 {
			optimise(&mut agents);

			totsum = 0.0;
			maxsum = 0.0;
			for agent in &agents {
				totsum += 1.0/agent.toterr;
				maxsum += 1.0/agent.maxerr;
			}

			let (maxerr, toterr) = (agents[0].maxerr, agents[0].toterr);
			let generation       = agents[0].brain.generation;
			println!("maxerr={maxerr:.2}, toterr={toterr:.2}, gen={generation}");

			// Quit training if things have started to converge
			if toterr == prverr {
				stayed += 1;
				if stayed > 23 || toterr == 0.0 {
					println!("\nn={n}");
					break
				}
			} else {
				prverr = toterr;
				stayed = 0
			}
		}
	}

	// Print top agent
	agents.sort_by(|a, b| a.toterr.partial_cmp(&b.toterr).unwrap());
	print_agent(&mut agents[0], inputs, targets);

	Ok(())
}
