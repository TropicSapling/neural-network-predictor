use crate::{agent::*, input, output, trainer::InvErrSum};

pub fn test(agent: &mut Agent, errsum: &mut InvErrSum, data: &[f64]) {
	agent.maxerr = 0.0;
	agent.toterr = 0.0;
	for i in 0..INPS {
		run(agent, &data[i..i+INPS], data[i+INPS]);
	}

	errsum.maxerr += 1.0/agent.maxerr;
	errsum.toterr += 1.0/agent.toterr;
}

pub fn run(agent: &mut Agent, inp: &[f64], aim: f64) -> [f64; OUTS] {
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
		err1 = (predictions[1] - predictions[0] - aim).abs()
	}

	agent.brain.discharge();

	agent.runtime = time.elapsed();
	agent.toterr += err1;

	// If error was worse for this input, record that
	if err1 > agent.maxerr {
		agent.maxerr = err1
	}

	predictions
}
