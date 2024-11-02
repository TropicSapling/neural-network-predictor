use std::fmt;
use std::time::Duration;

use crate::helpers::*;

pub const INPS: usize = 32;
pub const OUTS: usize = 2;

macro_rules! arr {
	($elem:expr) => (core::array::from_fn(|_| $elem))
}

#[derive(Debug)]
pub struct Agent {
	pub brain  : Brain,
	pub maxerr : f64,

	pub runtime: Duration
}


////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////

// TODO: Consider having HashMaps of Neurons with IDs instead of arrays.
// - Would likely significantly simplify the code.

/// neurons_inp: 365 numbers (normalise to [-1, 1]? if so how?)
/// neurons_out: [bp, sp] (normalise to [-1, 1]? if so how?)
#[derive(Clone)]
pub struct Brain {
	neurons_inp: [Neuron; INPS],
	neurons_hid: Vec<Neuron>,
	neurons_out: [Neuron; OUTS],

	pub gen: isize
}

#[derive(Clone)]
pub struct Neuron {
	pub excitation    : f64,
	pub act_threshold : f64,

	pub prev_conn: Vec<usize>,
	pub next_conn: Vec<OutwardConn>,

	reachable: bool,
	inv_mut: usize
}

#[derive(Clone, Debug)]
pub struct OutwardConn {
	pub dest_index: usize,
	pub weight: f64,
	pub relu: bool
}


////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////


impl Agent {
	pub fn from(agents: &Vec<Agent>, maxsum: f64) -> Self {
		// Create entirely new agents the first two times
		if agents.len() < 2 {
			return Agent::with(Brain::new(1, 0))
		}

		// Select parents
		let parent1 = Agent::select(agents, |parent| parent.maxerr, maxsum);
		let parent2 = Agent::select(agents, |parent| parent.maxerr, maxsum);

		// Return child of both
		Agent::merge(parent1, parent2)
	}

	fn mutate(mut self) -> Self {
		self.brain.mutate();
		self
	}

	fn with(brain: Brain) -> Self {
		Agent {brain, maxerr: 0.0, runtime: Duration::new(0, 0)}
	}

	fn merge(parent1: &Self, parent2: &Self) -> Self {
		Agent::with(Brain::merge(&parent1.brain, &parent2.brain)).mutate()
	}

	fn select(agents: &Vec<Agent>, err: impl Fn(&Self) -> f64, errsum: f64) -> &Self {
		// Try selecting a fit agent
		for _ in 0..7 {
			for parent in agents {
				// See error_share_formula.PNG
				let share = (1.0/err(parent)).powf(4.0);
				if rand_range(0.0..1.0) < share/errsum {
					return parent
				}
			}
		}

		&agents[0] // first agent as backup (chance: 1/e^7 ~ 0.09%)
	}
}


////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////


impl Brain {
	pub fn input(&mut self) -> &mut [Neuron; INPS] {&mut self.neurons_inp}

	pub fn discharge(&mut self) {
		for neuron in &mut self.neurons_hid {
			neuron.excitation = 0.0
		}
	}

	pub fn update_neurons(&mut self) -> &mut [Neuron; OUTS] {
		for i in 0..self.neurons_inp.len() {
			self.neurons_inp[i].reachable = true; // input neurons always reachable
			self.update_neuron(i, true)
		}

		for i in 0..self.neurons_hid.len() {
			self.update_neuron(i, false)
		}

		&mut self.neurons_out
	}

	fn update_neuron(&mut self, i: usize, is_input: bool) {
		let neuron = match is_input {
			true => &mut self.neurons_inp[i],
			_    => &mut self.neurons_hid[i]
		};

		let excitation = neuron.excitation;
		// Reset excitation
		neuron.excitation = 0.0;

		// If neuron activated...
		if excitation >= neuron.act_threshold {
			// ... prepare all connections for activation
			let mut activations = vec![];
			for conn in &neuron.next_conn {
				activations.push(conn.clone())
			}

			// ... activate the connections
			for conn in &mut activations {
				let recv_neuron = if conn.dest_index < OUTS {
					&mut self.neurons_out[conn.dest_index]
				} else {
					&mut self.neurons_hid[conn.dest_index - OUTS]
				};

				//let prev_recv_excitation = recv_neuron.excitation;

				if conn.relu {
					recv_neuron.excitation += conn.weight * excitation
				} else {
					recv_neuron.excitation += conn.weight;

					// STDP (Spike-Timing-Dependent Plasticity)
					// TODO: maybe make more realistic?
					// NOTE: Having this on => network changes on every run!
					/*if prev_recv_excitation >= recv_neuron.act_threshold {
						// Receiver already has fired => weaken connection
						Neuron::expand_or_shrink(&mut conn.weight, -1.0)
					} else if recv_neuron.excitation >= recv_neuron.act_threshold {
						// Receiver firing thanks to this => strengthen connection
						Neuron::expand_or_shrink(&mut conn.weight, 1.0)
					}*/
				}

				recv_neuron.reachable = true
			}

			let neuron = match is_input {
				true => &mut self.neurons_inp[i],
				_    => &mut self.neurons_hid[i]
			};

			// ... and finally apply potential STDP changes
			neuron.next_conn = activations
		}
	}

	fn mutate(&mut self) {
		let mut new_neurons = 0;
		let mut new_conns   = 0;

		// Mutate input neurons
		let mut i = 0;
		while i < self.neurons_inp.len() {
			let neuron = &mut self.neurons_inp[i];

			neuron.mutate(&mut new_neurons, &mut new_conns);

			// Ensure there is always at least one outgoing connection left
			if neuron.next_conn.len() < 1 {
				self.connect(i)
			}

			i += 1
		}

		// Mutate hidden neurons
		let mut i = 0;
		while i < self.neurons_hid.len() {
			let neuron = &mut self.neurons_hid[i];

			neuron.mutate(&mut new_neurons, &mut new_conns);

			// Remove neuron if it has no outgoing connections
			if neuron.next_conn.len() < 1 {
				for prev in neuron.prev_conn.clone() {
					let conns = match prev {
						0..INPS => &mut self.neurons_inp[prev       ].next_conn,
						_       => &mut self.neurons_hid[prev - INPS].next_conn,
					};

					conns.retain(|conn| conn.dest_index != OUTS + i)
				}

				/*self.neurons_hid.swap_remove(i);

				let hids              = self.neurons_hid.len();
				let swapped_in_neuron = &mut self.neurons_hid[i];

				swapped_in_neuron.prev_conn.retain(|prev| *prev != i);
				for prev in &mut swapped_in_neuron.prev_conn {
					if *prev == INPS + hids {
						*prev = i
					}
				}

				for prev in swapped_in_neuron.prev_conn.clone() {
					let conns = match prev {
						0..INPS => &mut self.neurons_inp[prev       ].next_conn,
						_       => &mut self.neurons_hid[prev - INPS].next_conn,
					};

					conns.retain(|conn| conn.dest_index != OUTS + i)
				}*/
			}

			i += 1
		}

		// Mutate output neurons
		for neuron in &mut self.neurons_out {
			neuron.mutate(&mut new_neurons, &mut new_conns);

			// Ensure there is always at least one outgoing connection left
			if neuron.next_conn.len() < 1 {
				neuron.next_conn.push(OutwardConn::new(self.neurons_hid.len() + OUTS))
			}
		}

		// Add new hidden neurons
		for _ in 0..new_neurons {
			self.neurons_hid.push(Neuron::new());
			self.connect(INPS + self.neurons_hid.len() - 1);
		}

		// Add new outgoing connections
		let neurons = INPS + self.neurons_hid.len() + OUTS;
		for _ in 0..new_conns {
			self.connect(rand_range(0..neurons))
		}
	}

	fn connect(&mut self, i: usize) {
		let conn = OutwardConn::new(self.neurons_hid.len() + OUTS);
		let nout = INPS + self.neurons_hid.len();

		if i < nout {
			// Connect back to neuron #i
			match conn.dest_index {
				0..OUTS => self.neurons_out[conn.dest_index       ].prev_conn.push(i),
				_       => self.neurons_hid[conn.dest_index - OUTS].prev_conn.push(i)
			}
		}

		// Connect out from neuron #i
		match i {
			0..INPS       => self.neurons_inp[i       ].next_conn.push(conn),
			_ if i < nout => self.neurons_hid[i - INPS].next_conn.push(conn),
			_             => self.neurons_out[i - nout].next_conn.push(conn)
		}
	}

	fn new(n: usize, gen: isize) -> Self {
		Brain {
			neurons_inp: arr![Neuron::new()   ],
			neurons_hid: vec![Neuron::new(); n],
			neurons_out: arr![Neuron::new()   ],
			gen
		}
	}

	fn merge(brain1: &Self, brain2: &Self) -> Self {
		let minhid = brain1.neurons_hid.len().min(brain2.neurons_hid.len());
		let maxhid = brain1.neurons_hid.len().max(brain2.neurons_hid.len());

		let mut brain = Brain::new(maxhid, brain1.gen.max(brain2.gen) + 1);

		// Merge input neurons
		for i in 0..INPS {
			brain.neurons_inp[i].merge(&brain1.neurons_inp[i], &brain2.neurons_inp[i])
		}

		// Merge initial hidden neurons
		for i in 0..minhid {
			brain.neurons_hid[i].merge(&brain1.neurons_hid[i], &brain2.neurons_hid[i])
		}

		// Merge remaining hidden neurons
		let maxbrain = if brain1.neurons_hid.len() == maxhid {brain1} else {brain2};
		for i in minhid..maxhid {
			brain.neurons_hid[i] = maxbrain.neurons_hid[i].clone()
		}

		// Merge output neurons
		for i in 0..OUTS {
			brain.neurons_out[i].merge(&brain1.neurons_out[i], &brain2.neurons_out[i])
		}

		brain
	}
}

impl Neuron {
	fn new() -> Self {
		Neuron {
			excitation    : 0.0,
			act_threshold : 0.0,

			prev_conn: vec![],
			next_conn: vec![],

			reachable: false,
			inv_mut: 2
		}
	}

	// By default 11/89 if mutation of mutation rate or not
	fn should_mutate_mut(inv_mut: usize) -> bool {rand_range(0..=inv_mut.pow(3)) == 0}
	// By default 33/67 if mutation or not
	fn should_mutate_now(inv_mut: usize) -> bool {rand_range(0..=inv_mut) == 0}
	// Always 50/50 if expansion or shrinking
	fn should_expand_now() -> bool {rand_range(0..=1) == 0}

	fn mutate(&mut self, new_neuron_count: &mut usize, new_conn_count: &mut usize) {
		// Mutate neuron properties
		if Neuron::should_mutate_mut(self.inv_mut) {
			self.inv_mut.add_bounded([-1, 1][rand_range(0..=1)])}
		if Neuron::should_mutate_now(self.inv_mut) {
			self.act_threshold += [-1.0, 1.0][rand_range(0..=1)]}

		// Mutate outgoing connections
		for conn in &mut self.next_conn {
			if Neuron::should_mutate_now(self.inv_mut) {
				if rand_range(0..(2 + conn.weight.abs() as usize)) == 0 {
					// Sometimes flip weight
					conn.weight = -conn.weight
				} else {
					if Neuron::should_expand_now() {
						// Sometimes expand weight or other stuff
						match rand_range(0..3) {
							0 => Neuron::expand_or_shrink(&mut conn.weight, 1.0),
							1 => *new_conn_count   += 1,
							_ => *new_neuron_count += 1
						}
					} else {
						// Sometimes shrink weight (which can effectively remove)
						Neuron::expand_or_shrink(&mut conn.weight, -1.0)
					}
				}
			}
		}

		// Remove effectively dead connections
		self.next_conn.retain(|conn| conn.weight != 0.0);

		// Reset excitation
		self.excitation = 0.0;

		// Assume not reachable until proven otherwise
		self.reachable = false
	}

	fn merge(&mut self, neuron1: &Neuron, neuron2: &Neuron) {
		match rand_range(0..2) {
			0 => *self = neuron1.clone(),
			_ => *self = neuron2.clone()
		}
	}

	fn expand_or_shrink(state: &mut f64, change: f64) {
		// Move towards or away from a neutral state of 0
		if *state > 0.0 {
			*state += change;
			if *state < 0.0 {
				*state = 0.0
			}
		} else {
			*state -= change;
			if *state > 0.0 {
				*state = 0.0
			}
		}
	}
}

impl OutwardConn {
	fn new(recv_neuron_count: usize) -> Self {
		OutwardConn {
			dest_index: rand_range(0..recv_neuron_count),
			weight: [-1.0, 1.0][rand_range(0..=1)],
			relu: [false, true][rand_range(0..=1)]
		}
	}
}

impl fmt::Debug for Brain {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		let mut s = String::from("Brain {\n");

		s += "\tneurons_inp: [\n";
		/*for neuron in &self.neurons_inp {
			s += &format!("\t\t{neuron:#?},\n")
		}*/ s += &format!("\t\t... ({})\n", INPS);

		let (mut unreachables, mut inactives) = (0, 0);

		s += "\t],\n\n\tneurons_hid: [\n";
		//for (i, neuron) in self.neurons_hid.iter().enumerate() {
		for neuron in self.neurons_hid.iter() {
			if neuron.reachable {
				//s += &format!("\t\t#{}: {neuron:#?},\n", i + OUTS)
			} else {
				unreachables += 1;
				if neuron.next_conn.len() < 1 {
					inactives += 1
				}
			}
		} s += &format!("\t\t... ({})\n", self.neurons_hid.len());
		s += &format!("\n\t\tUNREACHABLES: {unreachables} (inactive: {inactives})\n");

		s += "\t],\n\n\tneurons_out: [\n";
		for (i, neuron) in self.neurons_out.iter().enumerate() {
			s += &format!("\t\t#{i}: {neuron:#?},\n")
		}

		write!(f, "{s}\t],\n\n\tgen: {},\n}}", self.gen)
	}
}

impl fmt::Debug for Neuron {
	// Print neuron debug info in a concise way
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		if !self.reachable {
			if self.next_conn.len() < 1 {
				write!(f, "➖ Neuron {{UNREACHABLE & INACTIVE}}")
			} else {
				write!(f, "➖ Neuron {{UNREACHABLE, conns={}}}", self.next_conn.len())
			}
		} else if self.next_conn.len() < 1 {
			write!(f, "➖ Neuron {{INACTIVE}}")
		} else {
			let (is_at, act_at) = (self.excitation, self.act_threshold);

			// Mark firing neurons (red = negative response, green = positive)
			let s = String::from(
				if is_at >= act_at {
					let mut total_res = 0.0;
					for conn in &self.next_conn {
						total_res += conn.weight
					}

					if total_res < 0.0 {"🔴 "} else {"🟢 "}
				} else {"✖️ "}
			);

			let mut s = format!(
				"{s}Neuron {{IS@{:.1} | ACT@{:.1} | INVMUT@{} | ",
				             is_at,     act_at,     self.inv_mut
			);

			let mut conn_iter = self.next_conn.iter().peekable();
			while let Some(conn) = conn_iter.next() {
				let relu = if conn.relu {"*"} else {""};

				s += &format!("({relu}{:.1})->#{}", conn.weight, conn.dest_index);
				if !conn_iter.peek().is_none() {
					s += ", "
				}
			}

			write!(f, "{s}}}")
		}
	}
}
