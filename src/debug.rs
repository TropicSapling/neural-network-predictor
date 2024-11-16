use std::io::{stdout, Write};
use crate::{agent::Agent, ai, DATA_SIZE, INPS};

pub fn result(agent: &mut Agent, data: [f64; DATA_SIZE]) {
	agent.maxerr = 0.0;
	for i in 0..(DATA_SIZE - INPS - 1) {
		if i == DATA_SIZE - INPS*2 {
			println!("\n    ======================================================\n")
		}

		let inp = format!("#{i:<3} - {:<6.2} .. {:<6.2}", data[i], data[i+INPS-1]);
		let tgt = &data[i+INPS..i+INPS+2];
		let out = ai::run(agent, &data[i..i+INPS], tgt);

		let err = (out[0] - tgt[0]).abs().max((out[1] - tgt[1]).abs());

		println!("{inp} => {out:>6.2?} => maxerr={err:<4.2} <= {tgt:.2?}")
	}

	println!("\nNeural Network: {:#?}\n\nmaxerr={}", agent.brain, agent.maxerr);
}

pub fn progress(agent: &Agent, alive: usize, n: usize, iters: usize) {
	let maxerr = agent.maxerr;
	let gen    = agent.brain.gen;
	let t      = agent.runtime;

	let pb = format!("[{}>{}]", "=".repeat(n/(iters/26)), " ".repeat(26-n/(iters/26)));
	let st = format!("maxerr={maxerr:.2}, time={t:?}, gen={gen}");

	print!("\r{st:<36} {pb} (agents: {alive})");
	stdout().flush().unwrap();
}
