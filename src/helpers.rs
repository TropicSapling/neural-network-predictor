use std::ops::RangeBounds;
use rand::{Rng, distributions::uniform::{SampleRange, SampleUniform}};

pub trait BoundedSignedAdd {
	fn add_bounded(&mut self, val: isize);
}

impl BoundedSignedAdd for usize {
	fn add_bounded(&mut self, val: isize) {
		if val < 0 {
			*self = self.saturating_sub(-val as usize)
		} else {
			*self = self.saturating_add(val as usize)
		}
	}
}

pub fn rand_range<T, R>(range: R) -> T
	where T: SampleUniform,
	      R: RangeBounds<T> + SampleRange<T>
{
	rand::thread_rng().gen_range(range)
}
