use crate::{helpers::*, structs::*};

pub fn update_ai(agents: &mut Vec<Agent>, target: f64) {
	let (mut predictions, mut inverr, mut invsum) = (vec![], vec![], 0.0);
	for i in 0..agents.len() {
		let agent = &mut agents[i];
		let input = agent.brain.input();

		// INPUT
		for inp in input {
			// Set input to always fire
			(inp.excitation, inp.act_threshold) = (0.0, 0.0);

			// Set input weights
			for conn in &mut inp.next_conn {
				conn.weight = 123.0 // placeholder
			}
		}

		// INPUT -> ... -> OUTPUT
		let output = agent.brain.update_neurons();

		// OUTPUT
		predictions.push([0.0; OUTS]);
		for (n, out) in output.iter().enumerate() {
			if out.excitation >= out.act_threshold {
				for conn in &out.next_conn {
					predictions[i][n] += conn.weight
				}
			}
		}

		// Calculate absolute error...
		let error = (predictions[i][1] - predictions[i][0] - target).abs();
		// ... and record its inverse
		inverr.push(1.0/error);
		invsum += 1.0/error
	}

	// Prioritise spawning from existing generations
	for i in 0..inverr.len() {
		// See error_share_formula.PNG
		let share = inverr[i]/invsum;
		if rand_range(0.0..=invsum) < share {
			println!("{}", invsum/(inverr.len() as f64));
			agents.push(agents[i].spawn_child());
			break
		}
	}

	// But sometimes spawn an entirely new agent
	agents.push(Agent::new())
}
