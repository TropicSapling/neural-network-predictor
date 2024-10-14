use crate::agent::*;

pub fn assign(predictions: &mut [f64; OUTS], output: &mut [Neuron; OUTS]) {
	for (n, out) in output.iter_mut().enumerate() {
		predictions[n] = 0.0;

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

pub fn targets() -> [f64; INPS*2] {
	let mut arr = [0.0; INPS*2];
	for i in 0..INPS*2 {
		arr[i] = 3.0*(i as f64) + 14.0 // placeholder
	}
	arr
}
