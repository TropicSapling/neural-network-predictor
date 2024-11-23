use indexmap::map::Slice;
use crate::{agent::*, data::*};

pub fn assign(input: &mut Slice<usize, Neuron>, inp: &[DataRow]) {
	for (i, val) in inp.iter().flatten().enumerate() {
		input[i].excitation = val * crate::RESOLUTION // upscale by `RESOLUTION`
	}
}
