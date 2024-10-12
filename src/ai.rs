use crate::{agent::*, input, output};

pub fn update_ai(agent: &mut Agent, inp: f64, aim: f64) {
	// TODO: If get poor results, try pseudo-normalize i/o using log(...)
	// - Could potentially have "log" variant for each neuron
	// - ALSO: maybe try backpropagation?
	// - (additionally could try disabling halve-doubling of weights stuff)

	let mut err0 = 1.0;
	let mut err1 = 0.0;
	// Run until we get a "final" output error
	for _ in 0..4 {
		if err1 == err0 {break}

		let mut predictions = [0.0; OUTS];

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

	// If error was worse for this input, record that
	if err1 > agent.maxerr {
		agent.maxerr = err1
	}
}
