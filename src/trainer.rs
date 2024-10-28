use crate::{agent::Agent, ai, debug};
use crate::{INPS, DATA_SIZE, PARTITIONS};

struct Trainer {
	data: [f64; DATA_SIZE],

	partit: usize,
	maxsum: f64
}


////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////


impl Trainer {
	fn from(data: [f64; DATA_SIZE]) -> Self {
		Trainer {
			data,

			partit: 0,
			maxsum: 0.0
		}
	}

	fn data(&self) -> &[f64] {
		&self.data[self.partit*INPS..(self.partit+3)*INPS]
	}

	fn sort(agents: &mut Vec<Agent>) {
		let rank = |agent: &Agent| (agent.maxerr, -agent.brain.gen, agent.runtime);

		agents.sort_by(|a, b| rank(a).partial_cmp(&rank(b)).unwrap())
	}

	fn optimise(&self, agents: &mut Vec<Agent>) {
		Self::sort(agents);
		agents.truncate(128);
	}

	fn validate(&mut self, agents: &mut Vec<Agent>) -> f64 {
		let mut maxsum = 0.0;

		self.partit += 1;
		for agent in agents {
			maxsum += ai::test(agent, self.data())
		}
		self.partit -= 1;

		maxsum
	}

	fn crossval(&mut self, agents: &mut Vec<Agent>) {
		// Backup training errors
		let mut train_errs = vec![];
		for agent in &mut *agents {
			train_errs.push(agent.maxerr)
		}

		// Run against validation set (cross-validation)
		let val_maxsum = self.validate(agents);

		if agents[0].maxerr > train_errs[0] {
			// Switch training set if performance was poor
			self.partit = (self.partit + 1) % PARTITIONS;
			self.maxsum = val_maxsum;

			Self::sort(agents)
		} else {
			// Otherwise restore training errors
			self.maxsum = 0.0;
			for (i, agent) in agents.iter_mut().enumerate() {
				agent.maxerr = train_errs[i];
				self.maxsum += 1.0/agent.maxerr;
			}
		}
	}
}


////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////


pub fn train(agents: &mut Vec<Agent>, data: [f64; DATA_SIZE], iterations: usize) {
	let mut trainer = Trainer::from(data);

	for n in 1..=iterations {
		let mut agent = Agent::from(&agents, trainer.maxsum);

		// Train the agent...
		trainer.maxsum += ai::test(&mut agent, trainer.data());
		// ... and save it
		agents.push(agent);

		// Once in a while, prune and validate
		if n % 256 == 0 {
			trainer.optimise(agents);
			trainer.crossval(agents);

			debug::progress(&agents[0], n, iterations)
		}

		// TODO: Replace pruning with randomly selecting an agent for potential removal
		// - Chance of removal: `rand_range(0..inverr) == 0` except not quite
		// - Try save a "maximum" inverr

		if n % 8192 == 0 {println!("")}
	}

	println!("\n\nn={iterations}\n")
}
