use crate::{agent::*, input, output};

pub fn test(agent: &mut Agent, data: &[f64]) -> f64 {
	agent.maxerr = 0.0;
	for i in 0..INPS {
		run(agent, &data[i..i+INPS], &data[i+INPS..i+INPS+2]);
	}

	(1.0/agent.maxerr).powf(4.0)
}

pub fn run(agent: &mut Agent, inp: &[f64], aim: &[f64]) -> [f64; OUTS] {
	// TODO: If get poor results, try pseudo-normalize i/o using log(...)
	// - Could potentially have "log" variant for each neuron
	// - ALSO: maybe try backpropagation?
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
		err1 = (predictions[0] - aim[0]).abs().max((predictions[1] - aim[1]).abs())
	}

	agent.runtime = time.elapsed();
	agent.brain.discharge();

	// If error was worse for this input, record that
	if err1 > agent.maxerr {
		agent.maxerr = err1
	}

	predictions
}
