use crate::agent::*;

pub fn assign(input: &mut [Neuron; INPS]) {
	for inp in input {
		inp.excitation = 1.0 // placeholder
	}
}
