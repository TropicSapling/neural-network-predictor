mod helpers;

mod ai;
mod agent;
mod input;
mod output;

use agent::*;
use ai::update_ai;

fn print_agent(agent: &mut Agent, inputs: [f64; INPS*4], targets: &[f64]) {
	let (brain, maxerr, toterr) = (&agent.brain, agent.maxerr, agent.toterr);

	println!("\nNeural Network: {brain:#?}\n\nmaxerr={maxerr}\ntoterr={toterr}\n");

	agent.toterr = 0.0;
	agent.maxerr = 0.0;
	for i in 0..INPS*3 {
		if i == INPS*2 {
			println!("\n    ======================================================\n")
		}

		let inp = format!("{:<3.2} .. {:<3.2}", inputs[i], inputs[i+INPS-1]);
		let tgt = targets[i];
		let out = update_ai(agent.reset(), &inputs[i..i+INPS], tgt);

		let res = out[1] - out[0];
		let err = (res - tgt).abs();

		println!("{inp} => {out:>6.1?} => {res:<6.2} (err={err:<5.2}) <= {tgt:.2}")
	}
}

fn optimise(agents: &mut Vec<Agent>) {
	let rank = |agent: &Agent| agent.maxerr.powf(2.0) + agent.toterr;

	agents.sort_by(|a, b| rank(a).partial_cmp(&rank(b)).unwrap());
	agents.truncate(192);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let inputs  = input::inputs()?;
	let targets = &inputs[INPS..INPS*4];

	let mut agents: Vec<Agent> = vec![];
	let mut totsum = 0.0;
	let mut maxsum = 0.0;
	let mut prverr = [0.0; 2];
	let mut stayed = 0;
	let mut partit = 0;
	for n in 0..65536 {
		agents.push(Agent::from(&agents, totsum, maxsum));

		let agent = agents.last_mut().unwrap();

		agent.toterr = 0.0;
		agent.maxerr = 0.0;
		for i in partit*INPS..(partit+1)*INPS {
			update_ai(agent.reset(), &inputs[i..i+INPS], targets[i]);
		}

		totsum += 1.0/agent.toterr;
		maxsum += 1.0/agent.maxerr;

		// Remove worse-performing majority of agents once in a while
		if n % 384 == 0 {
			optimise(&mut agents);

			let (maxerr, toterr) = (agents[0].maxerr, agents[0].toterr);
			let generation       = agents[0].brain.generation;
			println!("maxerr={maxerr:.2}, toterr={toterr:.2}, gen={generation}");

			// Quit training if things have started to converge
			if toterr == prverr[partit] {
				stayed += 1;
				if stayed > 23 {
					println!("\nn={n}");
					break
				}
			} else {
				prverr[partit] = toterr;
				stayed         = 0
			}

			// Change input & target partitions - rerun for existing agents
			partit = (partit + 1) % 2;
			totsum = 0.0;
			maxsum = 0.0;
			for agent in &mut agents {
				agent.toterr = 0.0;
				agent.maxerr = 0.0;
				for i in partit*INPS..(partit+1)*INPS {
					update_ai(agent.reset(), &inputs[i..i+INPS], targets[i]);
				}

				totsum += 1.0/agent.toterr;
				maxsum += 1.0/agent.maxerr;
			}
		}
	}

	// Print top agent
	agents.sort_by(|a, b| a.toterr.partial_cmp(&b.toterr).unwrap());
	print_agent(&mut agents[0], inputs, targets);

	Ok(())
}
