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
// Partitions for cross-validation
const PARTITIONS: usize = 4;

fn print_agent(agent: &mut Agent, data: [f64; INPS*6]) {
	let (brain, maxerr, toterr) = (&agent.brain, agent.maxerr, agent.toterr);

	println!("\nNeural Network: {brain:#?}\n\nmaxerr={maxerr}\ntoterr={toterr}\n");

	agent.toterr = 0.0;
	agent.maxerr = 0.0;
	for i in 0..INPS*(PARTITIONS+1) {
		if i == INPS*PARTITIONS {
			println!("\n    ======================================================\n")
		}

		let inp = format!("#{i:<3} - {:<6.2} .. {:<6.2}", data[i], data[i+INPS-1]);
		let tgt = data[i+INPS];
		let out = update_ai(agent, &data[i..i+INPS], tgt);

		let res = out[1] - out[0];
		let err = (res - tgt).abs();

		println!("{inp} => {out:>6.2?} => {res:<6.2} (err={err:<4.2}) <= {tgt:.2}")
	}
}

fn printdbg(agent: &Agent, n: usize) {
	let (maxerr, toterr) = (agent.maxerr, agent.toterr);
	let gen              = agent.brain.gen;
	let t                = agent.runtime;

	let pb = format!("[{}>{}]", "=".repeat(n/1024), " ".repeat(26-n/1024));
	let st = format!("maxerr={maxerr:.2}, toterr={toterr:.2}, time={t:?}, gen={gen}");

	print!("\r{st:<52} {pb}");
	stdout().flush().unwrap();
}

fn rank(agent: &Agent) -> (f64, f64, isize, Duration) {
	(agent.maxerr, agent.toterr, -(agent.brain.gen as isize), agent.runtime)
}

fn optimise(agents: &mut Vec<Agent>) {
	agents.sort_by(|a, b| rank(a).partial_cmp(&rank(b)).unwrap());
	agents.truncate(128);
}

fn validate(agents: &mut Vec<Agent>, data: &[f64; INPS*6], p: usize) -> InvErrSum {
	let mut errsum = InvErrSum::new();
	for agent in agents {
		update(agent, &mut errsum, &data[p*INPS..(p+2)*INPS]);
	}

	errsum
}

fn update(agent: &mut Agent, errsum: &mut InvErrSum, data: &[f64]) {
	agent.maxerr = 0.0;
	agent.toterr = 0.0;
	for i in 0..INPS {
		update_ai(agent, &data[i..i+INPS], data[i+INPS]);
	}

	errsum.maxerr += 1.0/agent.maxerr;
	errsum.toterr += 1.0/agent.toterr;
}

struct InvErrSum {
	maxerr: f64,
	toterr: f64
}

impl InvErrSum {
	fn new() -> Self {
		InvErrSum {maxerr: 0.0, toterr: 0.0}
	}
}

fn main() -> Result<(), Box<dyn Error>> {
	let data = input::inputs()?;

	println!("");

	let mut agents: Vec<Agent> = vec![];
	let mut errsum: InvErrSum  = InvErrSum::new();

	let mut partit = 0;
	for n in 1..=26624 {
		let mut agent = Agent::from(&agents, errsum.maxerr, errsum.toterr);

		update(&mut agent, &mut errsum, &data[partit*INPS..(partit+2)*INPS]);

		agents.push(agent);

		// Once in a while, prune and validate
		if n % 256 == 0 {
			optimise(&mut agents);

			// Backup training errors
			let mut train_errs = vec![];
			for agent in &agents {
				train_errs.push((agent.maxerr, agent.toterr))
			}

			// Run against validation set (cross-validation)
			let val_errsum = validate(&mut agents, &data, (partit + 1) % PARTITIONS);

			// Switch training set if performance was poor
			if agents[0].maxerr > train_errs[0].0 {
				partit = (partit + 1) % PARTITIONS;
				errsum = val_errsum;

				// Resort (only needed for printing top scoring agent later)
				agents.sort_by(|a, b| rank(a).partial_cmp(&rank(b)).unwrap())
			} else {
				// Restore training errors
				errsum = InvErrSum::new();
				for (i, agent) in agents.iter_mut().enumerate() {
					agent.maxerr = train_errs[i].0;
					agent.toterr = train_errs[i].1;
				}
			}

			// Print top agent scores
			printdbg(&agents[0], n)
		}

		if n % 8192 == 0 {println!("")}
	}

	println!("\n\nn=26624");

	// Print final top agent
	print_agent(&mut agents[0], data);

	Ok(())
}
