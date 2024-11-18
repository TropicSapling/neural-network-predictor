use crate::{ai, ai::Error, debug, helpers::rand_range};
use crate::{Agent, INPS, DATA_SIZE, PARTITIONS};

struct Trainer {
	data: [f64; DATA_SIZE],

	partit: usize,
	errsum: Error
}


////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////


impl Trainer {
	fn from(data: [f64; DATA_SIZE]) -> Self {
		Trainer {
			data,

			partit: 0,
			errsum: Error::new()
		}
	}

	fn rank(agent: &Agent) -> (f64, f64, isize, std::time::Duration) {
		(agent.error.max, agent.error.tot, -agent.brain.gen, agent.runtime)
	}

	fn data(&self) -> &[f64] {
		&self.data[self.partit*INPS..(self.partit+3)*INPS]
	}

	fn sort(agents: &mut Vec<Agent>) {
		agents.sort_by(|a, b| Self::rank(a).partial_cmp(&Self::rank(b)).unwrap())
	}

	fn optimise(&self, agents: &mut Vec<Agent>) {
		Self::sort(agents);
		agents.truncate(128);
	}

	fn validate(&mut self, agents: &mut Vec<Agent>) -> Error {
		let mut errsum = Error::new();

		self.partit += 1;
		for agent in agents {
			errsum += ai::test(agent, self.data())
		}
		self.partit -= 1;

		errsum
	}

	fn crossval(&mut self, agents: &mut Vec<Agent>) {
		// Backup training errors
		let mut train_errs = vec![];
		for agent in &*agents {
			train_errs.push(agent.error.clone())
		}

		// Run against validation set (cross-validation)
		let val_errsum = self.validate(agents);

		if agents[0].error.max > train_errs[0].max {
			// Switch training set if performance was poor
			self.partit = (self.partit + 1) % PARTITIONS;
			self.errsum = val_errsum;

			Self::sort(agents)
		} else {
			// Otherwise restore training errors
			self.errsum = Error::new();
			for (i, agent) in agents.iter_mut().enumerate() {
				agent.error = train_errs[i].clone();

				self.errsum += Error {
					max: 1.0/agent.error.max,
					tot: 1.0/agent.error.tot
				}
			}
		}
	}
}


////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////


pub fn train(agents: &mut Vec<Agent>, data: [f64; DATA_SIZE], iterations: usize) {
	let mut trainer = Trainer::from(data);

	//let mut maxerr = f64::MAX;
	for n in 1..=iterations {
		let mut agent = Agent::from(&agents, &trainer.errsum);

		// Train the agent...
		let err = ai::train(&mut agent, trainer.data());
		if agent.runtime.as_micros() > 15 {
			continue // discard slow agents
		}
		// ... add its error
		trainer.errsum += err;
		// ... and save it
		agents.push(agent);

		// Once in a while, prune and validate
		if n % 256 == 0 {
			trainer.optimise(agents);
			trainer.crossval(agents);

			debug::progress(&agents[0], agents.len(), n, iterations)
		}

		/*if maxerr == f64::MAX {
			maxerr = agents.last().unwrap().maxerr
		}

		for _ in 0..agents.len()/128 {
			// Randomly select an agent to potentially remove
			let i = rand_range(0..agents.len());
			if rand_range(0.0..1.0) < agents[i].maxerr/maxerr && agents.len() > 128 {
				agents.swap_remove(i);
			}
		}*/

		if n % 8192 == 0 {println!("")}
	}

	println!("\n\nn={iterations}\n")
}
