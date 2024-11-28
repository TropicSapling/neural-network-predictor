use crate::{agent::*, consts::*, data::*, input, output};

#[derive(Clone, Debug)]
pub struct Error {
	pub max: f64,
	pub tot: f64
}

impl Error {
	pub fn new() -> Self {
		Error {max: 0.0, tot: 0.0}
	}

	pub fn max() -> Self {
		Error {max: f64::MAX, tot: f64::MAX}
	}
}

impl std::ops::AddAssign for Error {
	fn add_assign(&mut self, other: Self) {
		*self = Self {
			max: self.max + other.max,
			tot: self.tot + other.tot,
		}
	}
}

////////////////////////////////////////////////////////////////

pub fn train(agent: &mut Agent, data: &[DataRow]) -> Error {
	agent.error = Error::new();
	for i in 0..TEST_SIZE {
		let inp = &data[i..i+INPS_SIZE];
		let tgt = data[i+INPS_SIZE];
		let res = run(agent, inp, tgt, true);

		agent.brain.backprop(res, tgt)
	}

	Error {
		max: 1.0/agent.error.max,
		tot: 1.0/agent.error.tot
	}
}

pub fn test(agent: &mut Agent, data: &[DataRow]) -> Error {
	agent.error = Error::new();
	for i in 0..TEST_SIZE {
		run(agent, &data[i..i+INPS_SIZE], data[i+INPS_SIZE], true);
	}

	Error {
		max: 1.0/agent.error.max,
		tot: 1.0/agent.error.tot
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
		// Record total & max errors
		agent.error.tot += err1;
		if err1 > agent.error.max {
			agent.error.max = err1
		}
	}

	predictions
}
