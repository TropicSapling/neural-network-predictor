use crate::{agent::*, consts::*, data::*, input, output};

#[derive(Clone, Debug)]
pub struct Error {
	pub max: f64,
	pub avg: f64
}

impl Error {
	pub fn new() -> Self {
		Error {max: 0.0, avg: 0.0}
	}

	pub fn max() -> Self {
		Error {max: f64::MAX, avg: f64::MAX}
	}
}

impl std::ops::AddAssign for Error {
	fn add_assign(&mut self, other: Self) {
		*self = Self {
			max: self.max + other.max,
			avg: self.avg + other.avg,
		}
	}
}

impl std::ops::SubAssign for Error {
	fn sub_assign(&mut self, other: Self) {
		*self = Self {
			max: self.max - other.max,
			avg: self.avg - other.avg,
		}
	}
}

////////////////////////////////////////////////////////////////

pub fn train(agent: &mut Agent, data: &[DataRow]) -> Error {
	agent.error = Error::new();
	for i in 0..data.len()-INPS_SIZE {
		let inp = &data[i..i+INPS_SIZE];
		let tgt = data[i+INPS_SIZE];
		let res = run(agent, inp, tgt, true);

		agent.brain.backprop(res, tgt)
	}
	agent.error.avg /= (data.len() - INPS_SIZE) as f64;

	Error {
		max: 1.0/agent.error.max,
		avg: 1.0/agent.error.avg
	}
}

pub fn test(agent: &mut Agent, data: &[DataRow]) -> Error {
	agent.error = Error::new();
	for i in 0..data.len()-INPS_SIZE {
		run(agent, &data[i..i+INPS_SIZE], data[i+INPS_SIZE], true);
	}
	agent.error.avg /= (data.len() - INPS_SIZE) as f64;

	Error {
		max: 1.0/agent.error.max,
		avg: 1.0/agent.error.avg
	}
}

////////////////////////////////////////////////////////////////

pub fn run(agent: &mut Agent, inp: &[DataRow], aim: DataRow, save: bool) -> DataRow {
	// TODO: If get poor results, try pseudo-normalize i/o using log(...)
	// - Could potentially have "log" variant for each neuron
	let mut predictions = [0.0; OUTS];

	let mut err0 = 1.0;
	let mut err1 = 0.0;
	// Run until we get a "final" output error
	let time = std::time::Instant::now();
	for _ in 0..16 {
		if err1 == err0 {break}

		// Input
		input::assign(agent.brain.input(), inp);

		// Input -> ... -> Output
		let output = agent.brain.update_neurons();

		// Output
		output::assign(&mut predictions, output);

		// Calculate absolute error
		err0 = err1;
		err1 = (predictions[0] - aim[0]).powf(2.0)+(predictions[1] - aim[1]).powf(2.0)
	}

	agent.runtime = time.elapsed();
	agent.brain.discharge();

	if save {
		// Record average & maximum errors
		agent.error.avg += err1;
		if err1 > agent.error.max {
			agent.error.max = err1
		}
	}

	predictions
}
