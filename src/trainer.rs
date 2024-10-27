use std::time::Duration;

use crate::{agent::Agent, ai, debug};
use crate::{INPS, DATA_SIZE, PARTITIONS};

pub struct InvErrSum {
	pub maxerr: f64,
	pub toterr: f64
}

impl InvErrSum {
	fn new() -> Self {
		InvErrSum {maxerr: 0.0, toterr: 0.0}
	}
}

////////////////////////////////////////////////////////////////

fn rank(agent: &Agent) -> (f64, f64, isize, Duration) {
	(agent.maxerr, agent.toterr, -(agent.brain.gen as isize), agent.runtime)
}

fn optimise(agents: &mut Vec<Agent>) {
	agents.sort_by(|a, b| rank(a).partial_cmp(&rank(b)).unwrap());
	agents.truncate(128);
}

fn validate(agents: &mut Vec<Agent>, data: &[f64; DATA_SIZE], p: usize) -> InvErrSum {
	let mut errsum = InvErrSum::new();
	for agent in agents {
		ai::test(agent, &mut errsum, &data[p*INPS..(p+3)*INPS]);
	}

	errsum
}

pub fn train(agents: &mut Vec<Agent>, data: [f64; DATA_SIZE], iterations: usize) {
	let mut errsum = InvErrSum::new();
	let mut partit = 0;
	for n in 1..=iterations {
		let mut agent = Agent::from(&agents, errsum.maxerr, errsum.toterr);

		ai::test(&mut agent, &mut errsum, &data[partit*INPS..(partit+3)*INPS]);

		agents.push(agent);

		// Once in a while, prune and validate
		if n % 256 == 0 {
			optimise(agents);

			// Backup training errors
			let mut train_errs = vec![];
			for agent in &mut *agents {
				train_errs.push((agent.maxerr, agent.toterr))
			}

			// Run against validation set (cross-validation)
			let val_errsum = validate(agents, &data, partit + 1);

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
			debug::progress(&agents[0], n, iterations)
		}

		if n % 8192 == 0 {println!("")}
	}

	println!("\n\nn={iterations}\n")
}
