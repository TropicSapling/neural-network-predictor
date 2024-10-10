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

pub fn targets() -> [f64; 13824] {
	let mut arr = [0.0; 13824];
	for i in 0..13824 {
		//arr[i] = crate::helpers::rand_range(-3141592653.5..3141592653.5)
		arr[i] = 3141592653.5 // placeholder
	}
	arr
}
