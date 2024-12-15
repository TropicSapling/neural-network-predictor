use std::{fmt, time::Duration};
use indexmap::{IndexMap, map::Slice};
use crate::{ai::Error, consts::*, data::*, helpers::*};

#[derive(Debug)]
pub struct Agent {
	pub brain: Brain,
	pub error: Error,

	pub runtime: Duration
}


////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////


// Maybe normalise I/O to [-1, 1]? If so how?
#[derive(Clone)]
pub struct Brain {
	neurons: IndexMap<usize, Neuron>,
	next_id: usize,

	pub gen: isize
}

#[derive(Clone)]
pub struct Neuron {
	pub excitation    : isize,
	pub act_threshold : isize,

	pub prev_conn: Vec<usize>,
	pub next_conn: Vec<OutwardConn>,

	typ: NeuronType
}

#[derive(Clone, Debug)]
pub struct OutwardConn {
	pub dst_id: usize,
	pub weight: isize,
	pub charge: isize,

	pub relu: bool
}

#[derive(Clone)]
enum NeuronType {
	INP,
	HID,
	OUT
}

enum Evolution {}


////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////


impl Agent {
	pub fn from(agents: &Vec<Agent>, errsum: &Error) -> Self {
		// Create entirely new agents the first two times
		if agents.len() < 2 {
			return Agent::with(Brain::new(INPS+OUTS, 0))
		}

		// Select parents
		let parent1 = Agent::select(agents, |parent| parent.error.max, errsum.max);
		let parent2 = Agent::select(agents, |parent| parent.error.avg, errsum.avg);

		// Return child of both
		Agent::merge(parent1, parent2)
	}

	fn mutate(mut self) -> Self {
		self.brain.mutate();
		self
	}

	fn with(brain: Brain) -> Self {
		Agent {brain, error: Error::new(), runtime: Duration::new(0, 0)}
	}

	fn merge(parent1: &Self, parent2: &Self) -> Self {
		Agent::with(Brain::merge(&parent1.brain, &parent2.brain)).mutate()
	}

	fn select(agents: &Vec<Agent>, err: impl Fn(&Self) -> f64, errsum: f64) -> &Self {
		// Try selecting a fit agent
		for _ in 0..7 {
			for parent in agents {
				// See error_share_formula.PNG
				let share = 1.0/err(parent);
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
	pub fn input(&mut self) -> &mut Slice<usize, Neuron> {
		self.neurons.get_range_mut(0..INPS).unwrap()
	}

	pub fn discharge(&mut self) {
		for neuron in self.neurons.values_mut() {
			neuron.excitation = 0
		}
	}

	pub fn update_neurons(&mut self) -> &mut Slice<usize, Neuron> {
		for i in 0..self.neurons.len() {
			if i >= INPS && i < INPS+OUTS {continue}

			self.update_neuron(i)
		}

		self.neurons.get_range_mut(INPS..INPS+OUTS).unwrap()
	}

	pub fn backprop(&mut self, outputs: DataRow, targets: DataRow) {
		for i in INPS..INPS+OUTS {
			self.rewind_neuron(i, outputs[i-INPS] - targets[i-INPS])
		}
	}

	fn update_neuron(&mut self, i: usize) {
		let neuron = &mut self.neurons[i];

		// Save excitation before reset
		let excitation = neuron.excitation;
		// Reset excitation
		neuron.excitation = 0;

		// If neuron activated...
		if excitation >= neuron.act_threshold {
			// ... prepare all connections for activation
			let mut activations = vec![];
			for conn in &neuron.next_conn {
				activations.push(conn.clone())
			}

			// ... activate the connections
			for conn in &mut activations {
				if let Some(recv_neuron) = self.neurons.get_mut(&conn.dst_id) {
					conn.charge = match conn.relu {
						true => conn.weight * excitation,
						_    => conn.weight
					};

					recv_neuron.excitation += conn.charge
				} else {
					conn.weight = 0 // disable connection if broken
				}
			}

			// ... and finally apply potential changes
			self.neurons[i].next_conn = activations
		}
	}

	fn rewind_neuron(&mut self, _i: usize, _err: f64) {
		/*let id = *self.neurons.get_index(i).unwrap().0;

		let mut j = 0;
		while j < self.neurons[i].prev_conn.len() {
			let prev = self.neurons[i].prev_conn[j];
			if let Some(neuron) = self.neurons.get_mut(&prev) {
				for conn in &mut neuron.next_conn {
					if conn.dst_id == id {
						let sign = match conn.weight {
							..0.0 => -conn.charge.signum(),
							_     =>  conn.charge.signum()
						};

						match conn.relu {
							true => conn.weight -= err.signum()*sign,
							_    => conn.weight -= err.signum()
						}
					}
				}

				j += 1
			} else {
				self.neurons[i].prev_conn.swap_remove(j);
			}
		}*/
	}

	fn mutate(&mut self) {
		// Mutate neurons
		for i in 0..self.neurons.len() {
			if self.neurons[i].mutate() {
				self.connect(*self.neurons.get_index(i).unwrap().0)
			}
		}

		// Mutate neuron count
		if Evolution::should_mutate_now() {
			if Evolution::should_expand_now() {
				// Sometimes create new hidden neuron
				self.neurons.insert(self.next_id, Neuron::new(NeuronType::HID));
				self.connect(self.next_id);
				self.next_id += 1
			} else {
				// Sometimes weaken a random connection

				let mut rand = rand_range(0..self.neurons.len());
				while self.neurons[rand].next_conn.len() < 1 {
					rand = rand_range(0..self.neurons.len())
				}

				let rand_weight = &mut self.neurons[rand].next_conn.rand().weight;

				Neuron::expand_or_shrink(rand_weight, -1)
			}
		}

		// Retain only I/O neurons and active hidden neurons
		self.neurons.retain(|id, neuron| *id<INPS+OUTS || neuron.next_conn.len() > 0)
	}

	fn connect(&mut self, id: usize) {
		let conn = OutwardConn::to(self.rand_id());

		// Connect back to neuron #i
		self.neurons.get_mut(&conn.dst_id).unwrap().prev_conn.push(id);

		// Connect out from neuron #i
		self.neurons.get_mut(&id).unwrap().next_conn.push(conn)
	}

	fn new(n: usize, gen: isize) -> Self {
		let mut brain = Brain {
			neurons: IndexMap::new(),
			next_id: n,
			gen
		};

		// Create neurons
		for id in 0..n {
			let typ = match id {
				0..INPS             => NeuronType::INP,
				_ if id < INPS+OUTS => NeuronType::OUT,
				_                   => NeuronType::HID
			};

			brain.neurons.insert(id, Neuron::new(typ));
		}

		// Connect neurons
		for id in 0..n {
			brain.connect(id)
		}

		brain
	}

	fn merge(brain1: &Self, brain2: &Self) -> Self {
		let minlen = brain1.neurons.len().min(brain2.neurons.len());
		let maxlen = brain1.neurons.len().max(brain2.neurons.len());

		let mut brain = Brain::new(maxlen, brain1.gen.max(brain2.gen) + 1);

		// Merge initial neurons
		for i in 0..minlen {
			brain.neurons[i].merge(&brain1.neurons[i], &brain2.neurons[i])
		}

		// Merge remaining neurons
		let maxbrain = if brain1.neurons.len() == maxlen {brain1} else {brain2};
		for i in minlen..maxlen {
			brain.neurons[i] = maxbrain.neurons[i].clone()
		}

		brain
	}

	fn rand_id(&self) -> usize {
		*self.neurons.get_index(rand_range(INPS..self.neurons.len())).unwrap().0
	}
}

impl Neuron {
	fn new(typ: NeuronType) -> Self {
		Neuron {
			excitation    : 0,
			act_threshold : 0,

			prev_conn: vec![],
			next_conn: vec![],

			typ
		}
	}

	fn mutate(&mut self) -> bool {
		let mut add_conn = false;

		// Mutate activation threshold
		if Evolution::should_mutate_now() {
			self.act_threshold += [-1, 1][rand_range(0..=1)]
		}

		// Mutate connection count
		if Evolution::should_mutate_now() {
			if Evolution::should_expand_now() {
				// Sometimes create a new connection
				add_conn = true
			} else if self.next_conn.len() > 0 {
				// Sometimes weaken an existing connection
				Self::expand_or_shrink(&mut self.next_conn.rand().weight, -1)
			}
		}

		// Mutate outgoing connections
		for conn in &mut self.next_conn {
			if Evolution::should_mutate_now() {
				if rand_range(0..(2 + conn.weight.abs() as usize)) == 0 {
					// Sometimes flip weight
					conn.weight = -conn.weight
				} else {
					if Evolution::should_expand_now() {
						// Sometimes expand weight
						Self::expand_or_shrink(&mut conn.weight, 1)
					} else {
						// Sometimes shrink weight (which can effectively remove)
						Self::expand_or_shrink(&mut conn.weight, -1)
					}
				}
			}
		}

		// Remove effectively dead connections
		self.next_conn.retain(|conn| conn.weight != 0);

		// Reset excitation
		self.excitation = 0;

		add_conn
	}

	fn merge(&mut self, neuron1: &Neuron, neuron2: &Neuron) {
		match rand_range(0..2) {
			0 => *self = neuron1.clone(),
			_ => *self = neuron2.clone()
		}
	}

	fn expand_or_shrink(state: &mut isize, change: isize) {
		// Move towards or away from a neutral state of 0
		if *state > 0 {
			*state += change;
			if *state < 0 {
				*state = 0
			}
		} else {
			*state -= change;
			if *state > 0 {
				*state = 0
			}
		}
	}
}

impl OutwardConn {
	fn to(dst_id: usize) -> Self {
		OutwardConn {
			dst_id,
			weight: [-1, 1][rand_range(0..=1)],
			charge: 0,

			relu: [false, true][rand_range(0..=1)]
		}
	}
}

impl Evolution {
	// 20/80 if mutation or not
	fn should_mutate_now() -> bool {rand_range(0..=4) == 0}
	// 50/50 if expansion or shrinking
	fn should_expand_now() -> bool {rand_range(0..=1) == 0}
}

impl fmt::Debug for Brain {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		let mut s = String::from("Brain {\n");

		let mut inactives = 0;

		s += "\tneurons: [\n";
		for (_id, neuron) in &self.neurons {
			//s += &format!("\t\t#{id}: {neuron:#?},\n");
			if neuron.next_conn.len() < 1
			|| (neuron.prev_conn.len() < 1 && neuron.act_threshold > 0) {
				inactives += 1
			}
		} s += &format!("\t\t... ({})\n", self.neurons.len());
		s += &format!("\n\t\t(inactive: {inactives})\n");

		write!(f, "{s}\t],\n\n\tgen: {},\n}}", self.gen)
	}
}

impl fmt::Debug for Neuron {
	// Print neuron debug info in a concise way
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		if self.next_conn.len() < 1
		|| (self.prev_conn.len() < 1 && self.act_threshold > 0) {
			write!(f, "➖ Neuron {{INACTIVE}}")
		} else {
			let (is_at, act_at) = (self.excitation, self.act_threshold);

			// Mark firing neurons (red = negative response, green = positive)
			let s = String::from(
				if is_at >= act_at {
					let mut total_res = 0;
					for conn in &self.next_conn {
						total_res += conn.weight
					}

					if total_res < 0 {"🔴 "} else {"🟢 "}
				} else {"✖️ "}
			);

			let t = match self.typ {
				NeuronType::INP => "INP",
				NeuronType::HID => "HID",
				NeuronType::OUT => "OUT"
			};

			let mut s = format!("{s}{t} Neuron {{IS@{is_at:.1} | ACT@{act_at:.1} | ");

			let mut conn_iter = self.next_conn.iter().peekable();
			while let Some(conn) = conn_iter.next() {
				let relu = if conn.relu {"*"} else {""};

				s += &format!("({relu}{:.1})->#{}", conn.weight, conn.dst_id);
				if !conn_iter.peek().is_none() {
					s += ", "
				}
			}

			write!(f, "{s}}}")
		}
	}
}
