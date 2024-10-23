use crate::agent::*;

pub fn assign(predictions: &mut [f64; OUTS], output: &mut [Neuron; OUTS]) {
	for (n, out) in output.iter_mut().enumerate() {
		predictions[n] = 0.0;

		let excitation = out.excitation;
		// Reset excitation
		out.excitation = 0.0;

		// If neuron activated...
		if excitation >= out.act_threshold {
			// ... activate the connections
			for conn in &out.next_conn {
				if conn.relu {
					predictions[n] += conn.weight * excitation
				} else {
					predictions[n] += conn.weight
				}
			}
		}

		predictions[n] /= crate::RESOLUTION // downscale by `RESOLUTION`
	}
}
