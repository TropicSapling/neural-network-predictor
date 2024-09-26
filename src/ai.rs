use crate::structs::*;

pub fn update_ai(agents: &mut Vec<Agent>) {
	for i in 0..agents.len() {
		let agent = &mut agents[i];
		let input = agent.brain.input();

		// Input neurons always fire
		(input[0].excitation, input[0].act_threshold) = (0.0, 0.0);
		(input[1].excitation, input[1].act_threshold) = (0.0, 0.0);
		(input[2].excitation, input[2].act_threshold) = (0.0, 0.0);

		// Inputs
		for inp in input {
			for conn in &mut inp.next_conn {
				conn.weight = 123.0; // placeholder
			}
		}

		// Input -> ... -> Output
		let output = agent.brain.update_neurons();

		// BP prediction output
		agent.predictions[0] = 0.0;
		if output[0].excitation >= output[0].act_threshold {
			for conn in &output[0].next_conn {
				agent.predictions[0] += conn.weight;
			}
		}

		// SP prediction output
		agent.predictions[1] = 0.0;
		if output[1].excitation >= output[1].act_threshold {
			for conn in &output[1].next_conn {
				agent.predictions[1] += conn.weight;
			}
		}
	}
}
