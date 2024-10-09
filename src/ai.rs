use crate::{agent::*, input, output};

pub fn update_ai(agent: &mut Agent, target: f64, invsum: &mut f64, h: &mut f64)
	-> Result<(), Box<dyn std::error::Error>>
{
	if agent.inverr == 0.0 {
		let mut err0 = 1.0;
		let mut err1 = 0.0;
		// Run until we get a "final" output error
		for _ in 0..4 {
			if err1 == err0 {break}

			let mut predictions = [0.0; OUTS];

			// Input
			input::assign(agent.brain.input())?;

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

		// Record and print if new highscore
		if agent.inverr > *h {
			*h = agent.inverr;
			println!("TOP-ERR={}, gen={}", 1.0 / *h, agent.brain.generation);
		}
	}

	Ok(())
}
