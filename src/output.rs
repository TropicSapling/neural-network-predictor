use indexmap::map::Slice;
use crate::{agent::*, consts::*, data::*};

pub fn assign(predictions: &mut DataRow, output: &mut Slice<usize, Neuron>) {
	for (id, out) in output {
		predictions[id - INPS] = 0.0;

		let excitation = out.excitation;
		// Reset excitation
		out.excitation = 0;

		// If neuron activated...
		if excitation >= out.act_threshold {
			// ... activate the connections
			for conn in &out.next_conn {
				if conn.relu {
					predictions[id - INPS] += (conn.weight * excitation) as f64
				} else {
					predictions[id - INPS] += conn.weight as f64
				}
			}
		}

		predictions[id - INPS] /= RESOLUTION // downscale by `RESOLUTION`
	}
}
