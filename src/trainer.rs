use crate::{agent::*, ai, ai::Error, data::*, debug};//, helpers::rand_range};
use crate::consts::*;

// TODO: also save top-train agent for dbg printing
struct Trainer {
	data: Data,

	partit: usize,
	errsum: Error,
	valerr: Error
}


////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////


impl Trainer {
	fn from(data: Data) -> Self {
		Trainer {
			data,

			partit: 0,
			errsum: Error::new(),
			valerr: Error::max()
		}
	}

	fn rank(agent: &Agent) -> (f64, f64, isize, std::time::Duration) {
		(agent.error.tot, agent.error.max, -agent.brain.gen, agent.runtime)
	}

	fn data(&self) -> &[DataRow] {
		let size = TEST_SIZE + INPS_SIZE;

		&self.data[self.partit*size..(self.partit+1)*size]
	}

	fn sort(agents: &mut Vec<Agent>) {
		agents.sort_by(|a, b| Self::rank(a).partial_cmp(&Self::rank(b)).unwrap())
	}

	fn best_of(agents: &Vec<Agent>) -> usize {
		agents.iter().enumerate()
			.min_by(|(_,a), (_,b)| Self::rank(a).partial_cmp(&Self::rank(b)).unwrap())
			.map(|(i,_)| i).unwrap()
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
		// Get top-performing agent
		let top_train = Self::best_of(agents);

		// Backup training errors
		let mut train_errs = vec![];
		for agent in &*agents {
			train_errs.push(agent.error.clone())
		}

		// Run against validation set (cross-validation)
		let val_errsum = self.validate(agents);

		// If got poor validation error...
		if agents[top_train].error.tot > self.valerr.tot {
			// ... discard all agents of this epoch
			agents.truncate(128);

			// ... and switch training set
			self.partit = (self.partit + 1) % PARTITIONS;
			self.errsum = val_errsum
		} else {
			// Otherwise, save new validation error...
			self.valerr = agents[top_train].error.clone();

			// ... and restore training errors
			for (agent, train_err) in agents.iter_mut().zip(train_errs) {
				agent.error = train_err
			}
		}
	}
}


////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////


pub fn train(agents: &mut Vec<Agent>, data: Data, iterations: usize) {
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

		// Once in a while, prune and validate (end epoch)
		if n % 256 == 0 {
			trainer.crossval(agents);
			trainer.optimise(agents);

			let agents_alive = agents.len();

			debug::progress(&mut agents[0], data, agents_alive, n, iterations)
		}

		/*if maxerr == f64::MAX {
			maxerr = agents.last().unwrap().error.max
		}

		for _ in 0..agents.len()/128 {
			// Randomly select an agent to potentially remove
			let i = rand_range(0..agents.len());
			if rand_range(0.0..1.0) < agents[i].error.max/maxerr && agents.len() > 8 {
				agents.swap_remove(i);
			}
		}*/

		if n % 8192 == 0 {println!("")}
	}

	println!("\n\nn={iterations}\n")
}
