use indexmap::map::Slice;
use crate::{agent::*, consts::*, data::*};

pub fn assign(input: &mut Slice<usize, Neuron>, inp: &[DataRow]) {
	for (i, val) in inp.iter().flatten().enumerate() {
		input[i].excitation = (val * RESOLUTION) as isize // upscale by `RESOLUTION`
	}
}
