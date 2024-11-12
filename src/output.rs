use indexmap::map::Slice;
use crate::agent::*;

pub fn assign(predictions: &mut [f64; OUTS], output: &mut Slice<usize, Neuron>) {
	for (id, out) in output {
		predictions[id - INPS] = 0.0;

		let excitation = out.excitation;
		// Reset excitation
		out.excitation = 0.0;

		// If neuron activated...
		if excitation >= out.act_threshold {
			// ... activate the connections
			for conn in &out.next_conn {
				if conn.relu {
					predictions[id - INPS] += conn.weight * excitation
				} else {
					predictions[id - INPS] += conn.weight
				}
			}
		}

		predictions[id - INPS] /= crate::RESOLUTION // downscale by `RESOLUTION`
	}
}
