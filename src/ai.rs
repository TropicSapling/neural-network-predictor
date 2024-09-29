use crate::structs::*;

pub fn update_ai(agent: &mut Agent, target: f64, invsum: &mut f64, h: &mut f64) {
	if agent.inverr == 0.0 {
		let mut err0 = 1.0;
		let mut err1 = 0.0;
		// Run until we get a "final" output error
		while err1 != err0 {
			let input = agent.brain.input();

			// INPUT
			for inp in input {
				// Set input to always fire
				(inp.excitation, inp.act_threshold) = (0.0, 0.0);

				// Set input weights
				for conn in &mut inp.next_conn {
					conn.weight = 4.0 // placeholder
				}
			}

			// INPUT -> ... -> OUTPUT
			let output = agent.brain.update_neurons();

			// OUTPUT
			let mut predictions = [0.0; OUTS];
			for (n, out) in output.iter().enumerate() {
				if out.excitation >= out.act_threshold {
					for conn in &out.next_conn {
						predictions[n] += conn.weight
					}
				}
			}

			// Calculate absolute error
			err0 = err1;
			err1 = (predictions[1] - predictions[0] - target).abs();
		}

		// Record error inverse
		agent.inverr = 1.0/err1;
		*invsum += agent.inverr;

		// Record and print if new highscore
		if agent.inverr > *h {
			*h = agent.inverr;
			println!("error={}", 1.0 / *h)
		}
	}
}
