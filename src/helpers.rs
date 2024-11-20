use std::ops::RangeBounds;
use rand::{Rng, distributions::uniform::{SampleRange, SampleUniform}};

pub fn rand_range<T, R>(range: R) -> T
	where T: SampleUniform,
	      R: RangeBounds<T> + SampleRange<T>
{
	rand::thread_rng().gen_range(range)
}

pub trait SliceRand<T> {
	fn rand(&mut self) -> &mut T;
}

impl<T> SliceRand<T> for [T] {
	fn rand(&mut self) -> &mut T {
		&mut self[rand_range(0..self.len())]
	}
}
