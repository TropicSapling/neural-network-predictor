use crate::{agent::*, input, output};

pub fn update_ai(agent: &mut Agent, target: f64, invsum: &mut f64, i: usize)
	-> Result<(), Box<dyn std::error::Error>>
{
	// TODO: if get poor results, try pseudo-normalize i/o using log(...)
	// - Could potentially have "log" variant for each neuron

	let mut err0 = 1.0;
	let mut err1 = 0.0;
	// Run until we get a "final" output error
	for _ in 0..4 {
		if err1 == err0 {break}

		let mut predictions = [0.0; OUTS];

		// Input
		input::assign(agent.brain.input(), i)?;

		// Input -> ... -> Output
		let output = agent.brain.update_neurons();

		// Output
		output::assign(&mut predictions, output);

		// Calculate absolute error
		err0 = err1;
		err1 = (predictions[1] - predictions[0] - target).abs()
	}

	// Record error inverse
	agent.inverr = 1.0/err1;
	*invsum += agent.inverr;

	Ok(())
}
