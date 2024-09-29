use std::fmt;
use crate::helpers::*;

pub const INPS: usize = 365;
pub const OUTS: usize = 2;

#[derive(Debug)]
pub struct Agent {
	pub brain  : Brain,
	pub inverr : f64
}


////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////


/// neurons_inp: 365 numbers (normalise to [-1, 1]? if so how?)
/// neurons_out: [bp, sp] (normalise to [-1, 1]? if so how?)
#[derive(Clone)]
pub struct Brain {
	neurons_inp: [Neuron; INPS],
	neurons_hid: Vec<Neuron>,
	neurons_out: [Neuron; OUTS],

	generation: usize // for debugging/display
}

#[derive(Clone)]
pub struct Neuron {
	pub excitation: f64,
	pub tick_drain: f64,

	pub act_threshold: f64,

	pub next_conn: Vec<OutwardConn>,

	reachable: bool
}

#[derive(Clone, Debug)]
pub struct OutwardConn {
	pub dest_index: usize,
	pub speed: usize, // currently unused
	pub weight: f64,

	relu: bool
}


////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////


impl Agent {
	pub fn new(agents: &Vec<Agent>, invsum: f64) -> Self {
		// Prioritise spawning from existing generations
		for i in 0..agents.len() {
			// See error_share_formula.PNG
			let share = agents[i].inverr / invsum;
			if rand_range(0.0..1.0) < share {
				return agents[i].spawn_child()
			}
		}

		// But sometimes spawn an entirely new agent
		let mut new_agent = Agent::with(Brain {
			neurons_inp: core::array::from_fn(|_| Neuron::new(6+OUTS)),
			neurons_hid: vec![
				Neuron::new(6+OUTS), Neuron::new(6+OUTS), Neuron::new(6+OUTS),
				Neuron::new(6+OUTS), Neuron::new(6+OUTS), Neuron::new(6+OUTS)
			],
			neurons_out: core::array::from_fn(|_| Neuron::new(6+OUTS)),
			generation: 0
		});

		for _ in 0..rand_range(0..8) {
			new_agent = new_agent.mutate()
		}

		new_agent
	}

	pub fn spawn_child(&self) -> Self {
		let mut brain = self.brain.clone();

		brain.generation += 1;

		// Spawn identical copy of self in 1/3 of cases, otherwise mutate
		return if rand_range(0..3) == 0 {
			Agent::with(brain)
		} else {
			Agent::with(brain).mutate()
		}
	}

	fn with(brain: Brain) -> Self {
		Agent {brain, inverr: 0.0}
	}

	fn mutate(mut self) -> Self {
		self.brain.mutate();
		self
	}
}


////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////


impl Brain {
	pub fn input(&mut self) -> &mut [Neuron; INPS] {&mut self.neurons_inp}

	pub fn update_neurons(&mut self) -> &[Neuron; OUTS] {
		// Drain output neurons from previous excitation
		for i in 0..OUTS {
			self.neurons_out[i].drain()
		}

		for i in 0..self.neurons_inp.len() {
			self.neurons_inp[i].reachable = true; // input neurons always reachable
			self.update_neuron(i, true)
		}

		for i in 0..self.neurons_hid.len() {
			if self.neurons_hid[i].reachable {
				self.update_neuron(i, false);
				self.neurons_hid[i].drain()
			}
		}

		&self.neurons_out
	}

	fn update_neuron(&mut self, i: usize, is_input: bool) {
		let neuron = match is_input {
			true => &self.neurons_inp[i],
			_    => &self.neurons_hid[i]
		};

		let excitation = neuron.excitation;

		// If neuron activated...
		if excitation >= neuron.act_threshold {
			// ... prepare all connections for activation
			let mut activations = vec![];
			for conn in &neuron.next_conn {
				activations.push(conn.clone())
			}

			// ... and then activate the connections
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
					/*if prev_recv_excitation >= recv_neuron.act_threshold {
						// Receiver already has fired => weaken connection
						if conn.weight.abs() > 1.0 {
							Neuron::expand_or_shrink(&mut conn.weight, -1.0)
						}
					} else if recv_neuron.excitation >= recv_neuron.act_threshold {
						// Receiver firing thanks to this => strengthen connection
						if conn.weight.abs() < 8.0 {
							Neuron::expand_or_shrink(&mut conn.weight, 1.0)
						}
					}*/
				}

				recv_neuron.reachable = true
			}

			// ... and finally apply potential STDP changes
			match is_input {
				true => self.neurons_inp[i].next_conn = activations,
				_    => self.neurons_hid[i].next_conn = activations
			}
		}
	}

	fn mutate(&mut self) {
		let mut recv_neurons = self.neurons_hid.len() + OUTS;
		let mut new_neurons  = 0;
		let mut new_conns    = 0;

		// Mutate input neurons
		for neuron in &mut self.neurons_inp {
			neuron.mutate(&mut new_neurons, &mut new_conns, recv_neurons)
		}

		// Mutate hidden neurons
		for neuron in &mut self.neurons_hid {
			neuron.mutate(&mut new_neurons, &mut new_conns, recv_neurons)
		}

		// Mutate output neurons
		for neuron in &mut self.neurons_out {
			neuron.mutate(&mut new_neurons, &mut new_conns, recv_neurons)
		}

		// Add new hidden neurons
		for _ in 0..new_neurons {
			self.neurons_hid.push(Neuron::new(recv_neurons));
			recv_neurons += 1
		}

		// Add new outgoing connections
		for _ in 0..new_conns {
			let hids = self.neurons_hid.len();
			let rand = rand_range(0..INPS+hids+OUTS);

			let neuron = if rand < INPS {
				&mut self.neurons_inp[rand]
			} else if rand < INPS+hids {
				&mut self.neurons_hid[rand-INPS]
			} else {
				&mut self.neurons_out[rand-INPS-hids]
			};

			neuron.next_conn.push(OutwardConn::new(recv_neurons))
		}
	}
}

impl Neuron {
	fn new(recv_neuron_count: usize) -> Self {
		Neuron {
			excitation: 0.0,
			tick_drain: 1.0,

			act_threshold: 0.0,

			next_conn: vec![OutwardConn::new(recv_neuron_count)],

			reachable: false
		}
	}

	// TODO: maybe have mutation rate part of neuron properties?

	// 33/67 if mutation or not
	fn should_mutate_now() -> bool {rand_range(0..3) == 0}
	// 67/33 if expansion or shrinking
	fn should_expand_now() -> bool {rand_range(0..3) < 2}

	fn mutate(&mut self,
		new_neuron_count  : &mut usize,
		new_conn_count    : &mut usize,
		recv_neuron_count :      usize
	) {
		// Mutate neuron properties
		if Neuron::should_mutate_now() {
			self.tick_drain += [-1.0, 1.0][rand_range(0..=1)]}
		if Neuron::should_mutate_now() {
			self.act_threshold += [-1.0, 1.0][rand_range(0..=1)]}

		// Mutate outgoing connections
		for conn in &mut self.next_conn {
			if Neuron::should_mutate_now() {
				if rand_range(0..(2 + conn.weight.abs() as usize)) == 0 {
					// Sometimes flip weight
					conn.weight = -conn.weight
				} else {
					if Neuron::should_expand_now() {
						// Sometimes expand weight or other stuff
						match rand_range(0..3) {
							0 => Neuron::expand_or_shrink(&mut conn.weight, 1.0),
							1 => *new_conn_count += 1,
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
		self.next_conn.retain(|conn| (conn.weight*10.0).round() != 0.0);

		// If this neuron is inactive, can be recycled
		if self.next_conn.len() < 1 && *new_neuron_count > 0 {
			*new_neuron_count -= 1;
			self.next_conn.push(OutwardConn::new(recv_neuron_count))
		}

		// Reset excitation
		self.excitation = 0.0;

		// Assume not reachable until proven otherwise
		self.reachable = false
	}

	fn drain(&mut self) {
		Neuron::expand_or_shrink(&mut self.excitation, -self.tick_drain.abs())
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
			speed: 0,
			weight: [-1.0, 1.0][rand_range(0..=1)],
			relu: [false, true][rand_range(0..=1)]
		}
	}
}

impl fmt::Debug for Brain {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		let mut s = String::from("Brain {\n");

		s += "\tneurons_inp: [\n";
		for neuron in &self.neurons_inp {
			s += &format!("\t\t{neuron:#?},\n")
		}

		let (mut unreachables, mut inactives) = (0, 0);

		s += "\t],\n\n\tneurons_hid: [\n";
		for (i, neuron) in self.neurons_hid.iter().enumerate() {
			if neuron.reachable {
				s += &format!("\t\t#{}: {neuron:#?},\n", i + OUTS)
			} else {
				unreachables += 1;
				if neuron.next_conn.len() < 1 {
					inactives += 1
				}
			}
		}
		s += &format!("\n\t\tUNREACHABLES: {unreachables} (inactive: {inactives})\n");

		s += "\t],\n\n\tneurons_out: [\n";
		for (i, neuron) in self.neurons_out.iter().enumerate() {
			s += &format!("\t\t#{i}: {neuron:#?},\n")
		}

		write!(f, "{s}\t],\n\n\tgeneration: {},\n}}", self.generation)
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
						total_res += conn.weight;  
					}

					if total_res < 0.0 {"🔴 "} else {"🟢 "}
				} else {"✖️ "}
			);

			let mut s = format!("{s}Neuron {{IS@{:.1} | ACT@{:.1} | ", is_at, act_at);

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
