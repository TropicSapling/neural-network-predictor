use crate::agent::*;

pub fn assign(predictions: &mut [f64; OUTS], output: &[Neuron; OUTS]) {
	for (n, out) in output.iter().enumerate() {
		if out.excitation >= out.act_threshold {
			for conn in &out.next_conn {
				if conn.relu {
					predictions[n] += conn.weight * out.excitation
				} else {
					predictions[n] += conn.weight
				}
			}
		}
	}
}

pub fn targets() -> [f64; 55296] {
	let mut arr = [0.0; 55296];
	for i in 0..55296 {
		arr[i] = crate::helpers::rand_range(-3141592653.5..3141592653.5) // placeholder
	}
	arr
}
