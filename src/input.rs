use crate::agent::*;

pub fn assign(input: &mut [Neuron; INPS], inp: &[f64]) {
	for i in 0..INPS {
		input[i].excitation = inp[i] * crate::RESOLUTION // upscale by `RESOLUTION`
	}
}
