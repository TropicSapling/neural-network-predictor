mod helpers;

mod ai;
mod agent;
mod input;
mod output;

use std::{error::Error, io::{stdout, Write}, time::Duration};

use agent::*;
use ai::update_ai;

// All I/O is upscaled/downscaled by 1000x
const RESOLUTION: f64 = 1000.0;

fn print_agent(agent: &mut Agent, inputs: [f64; INPS*4], targets: &[f64]) {
	let (brain, maxerr, toterr) = (&agent.brain, agent.maxerr, agent.toterr);

	println!("\nNeural Network: {brain:#?}\n\nmaxerr={maxerr}\ntoterr={toterr}\n");

	agent.toterr = 0.0;
	agent.maxerr = 0.0;
	for i in 0..INPS*3 {
		if i == INPS*2 {
			println!("\n    ======================================================\n")
		}

		let inp = format!("{:<5.2} .. {:<5.2}", inputs[i], inputs[i+INPS-1]);
		let tgt = targets[i];
		let out = update_ai(agent.reset(), &inputs[i..i+INPS], tgt);

		let res = out[1] - out[0];
		let err = (res - tgt).abs();

		println!("{inp} => {out:>5.2?} => {res:<5.2} (err={err:<4.2}) <= {tgt:.2}")
	}
}

fn printdbg(agents: &Vec<Agent>, n: usize) {
	let (maxerr, toterr) = (agents[0].maxerr, agents[0].toterr);
	let gen              = agents[0].brain.generation;
	let t                = agents[0].runtime;

	let pb = format!("[{}>{}]", "=".repeat(n/2048), " ".repeat(26-n/2048));
	let st = format!("maxerr={maxerr:.2}, toterr={toterr:.2}, time={t:?}, gen={gen}");

	print!("\r{st:<47} {pb}");
	stdout().flush().unwrap();
}

fn rank(agent: &Agent) -> (f64, f64, isize, Duration) {
	(agent.maxerr, agent.toterr, -(agent.brain.generation as isize), agent.runtime)
}

fn optimise(agents: &mut Vec<Agent>) {
	agents.sort_by(|a, b| rank(a).partial_cmp(&rank(b)).unwrap());
	agents.truncate(128);
}

fn main() -> Result<(), Box<dyn Error>> {
	println!("");

	let inputs  = input::inputs()?;
	let targets = &inputs[INPS..INPS*4];

	let mut agents: Vec<Agent> = vec![];
	let mut totsum = 0.0;
	let mut maxsum = 0.0;
	let mut prverr = [(0.0, 0.0); 2];
	let mut bsterr = (f64::MAX, f64::MAX);
	let mut partit = 0;
	for n in 1..=53248 {
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
		if n % 256 == 0 {
			optimise(&mut agents);
			printdbg(&agents, n);

			// Quit training if things have started to converge
			prverr[partit] = (agents[0].toterr, agents[0].maxerr);

			let mut worst_err: (f64, f64) = (0.0, 0.0);
			for err in prverr {
				worst_err.0 = worst_err.0.max(err.0);
				worst_err.1 = worst_err.1.max(err.1);
			}

			if worst_err.0 < bsterr.0/4.0 || worst_err.1 < bsterr.1/4.0 {
				bsterr = worst_err;
				if n > 2048 {println!("")}
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

	println!("\n\nn=53248");

	// Print top agent
	agents.sort_by(|a, b| a.toterr.partial_cmp(&b.toterr).unwrap());
	print_agent(&mut agents[0], inputs, targets);

	Ok(())
}
