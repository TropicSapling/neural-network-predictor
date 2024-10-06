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
