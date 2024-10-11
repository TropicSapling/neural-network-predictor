use crate::agent::*;

pub fn assign(predictions: &mut [f64; OUTS], output: &mut [Neuron; OUTS]) {
	for (n, out) in output.iter_mut().enumerate() {
		// If neuron activated...
		if out.excitation >= out.act_threshold {
			// ... activate the connections
			for conn in &out.next_conn {
				if conn.relu {
					predictions[n] += conn.weight * out.excitation
				} else {
					predictions[n] += conn.weight
				}
			}

			// ... and reset excitation
			out.excitation = 0.0
		}
	}
}

pub fn targets() -> [f64; 55296] {
	let mut arr = [0.0; 55296];
	for i in 0..55296 {
		arr[i] = 314.15926535 + (i as f64)%52.0 // placeholder
	}
	arr
}
