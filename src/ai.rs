use crate::structs::*;

pub fn update_ai(agents: &mut Vec<Agent>) {
	for i in 0..agents.len() {
		let agent = &mut agents[i];
		let input = agent.brain.input();

		// Input neurons always fire
		(input[0].excitation, input[0].act_threshold) = (0.0, 0.0);
		(input[1].excitation, input[1].act_threshold) = (0.0, 0.0);
		(input[2].excitation, input[2].act_threshold) = (0.0, 0.0);

		// Relative size of nearest as first input
		for conn in &mut input[0].next_conn {
			conn.weight = if false {
				1.0
			} else if true {
				-1.0
			} else {0.0}
		}

		// Distance to nearest as second input
		for conn in &mut input[1].next_conn {
			conn.weight = 123.0
		}

		// Angle towards nearest as third input
		for conn in &mut input[2].next_conn {
			conn.weight = 456.0
		}

		// Input -> ... -> Output
		let output = agent.brain.update_neurons();

		// Movement output
		//body.mov = 0.0;
		if output[0].excitation >= output[0].act_threshold {
			for conn in &output[0].next_conn {
				//body.mov += conn.weight;
			}
		}

		// Rotation output
		//body.rot = 0.0;
		if output[1].excitation >= output[1].act_threshold {
			for conn in &output[1].next_conn {
				//body.rot += conn.weight;
			}
		}
	}
}
