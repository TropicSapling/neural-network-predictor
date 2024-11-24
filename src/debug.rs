use std::io::{stdout, Write};
use crate::{ai, ai::*, agent::*, consts::*, data::*};

pub fn result(agent: &mut Agent, data: Data) {
	let mut error = Error::new();
	for i in 0..data.len()-INPS_SIZE {
		if i == data.len() - 2*INPS_SIZE {
			print!("\n                   ");
			println!("===========================================================\n");
		}

		// Run agent
		let tgt = data[i+INPS_SIZE];
		let out = ai::run(agent, &data[i..i+INPS_SIZE], tgt);

		// Calculate error
		let err = (out[0] - tgt[0]).abs() + (out[1] - tgt[1]).abs();

		// Record total & max errors
		error.tot += err;
		if err > error.max {
			error.max = err
		}

		// Format input
		let j   = i + INPS_SIZE - 1;
		let inp = format!("{i:0>3}..{j:0>3} - {:>6.2?}..{:>6.2?}", data[i], data[j]);

		// Print results for this input
		println!("{inp} => {out:>6.2?} (err={err:0>5.2}) <= {tgt:>6.2?}")
	}

	println!("\nNeural Network: {:#?}\n\n{error:.2?}", agent.brain);
}

pub fn progress(agent: &Agent, alive: usize, n: usize, iters: usize) {
	let maxerr = agent.error.max.sqrt();
	let toterr = (agent.error.tot/(INPS as f64)).sqrt() * (INPS as f64);
	let gen    = agent.brain.gen;
	let t      = agent.runtime;

	let pb = format!("[{}>{}]", "=".repeat(n/(iters/26)), " ".repeat(26-n/(iters/26)));
	let st = format!("maxerr={maxerr:.2}, toterr={toterr:.2}, time={t:?}, gen={gen}");

	print!("\r{st:<50} {pb} (agents: {alive})");
	stdout().flush().unwrap();
}
