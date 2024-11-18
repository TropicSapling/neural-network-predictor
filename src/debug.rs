use std::io::{stdout, Write};
use crate::{ai, ai::Error, Agent, DATA_SIZE, INPS};

pub fn result(agent: &mut Agent, data: [f64; DATA_SIZE]) {
	let mut error = Error::new();
	for i in 0..(DATA_SIZE - INPS - 1) {
		if i == DATA_SIZE - INPS*2 {
			println!("\n    ======================================================\n")
		}

		// Run agent
		let inp = format!("#{i:<3} - {:<6.2} .. {:<6.2}", data[i], data[i+INPS-1]);
		let tgt = &data[i+INPS..i+INPS+2];
		let out = ai::run(agent, &data[i..i+INPS], tgt);

		// Calculate error
		let err = (out[0] - tgt[0]).abs() + (out[1] - tgt[1]).abs();

		// Record total & max errors
		error.tot += err;
		if err > error.max {
			error.max = err
		}

		// Print results for this input
		println!("{inp} => {out:>6.2?} (err={err:<4.2}) <= {tgt:.2?}")
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
