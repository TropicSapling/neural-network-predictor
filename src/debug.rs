use std::io::{stdout, Write};
use crate::{ai, ai::Error, Agent, DATA_SIZE, INPS};

pub fn result(agent: &mut Agent, data: [f64; DATA_SIZE]) {
	agent.error = Error::new();
	for i in 0..(DATA_SIZE - INPS - 1) {
		if i == DATA_SIZE - INPS*2 {
			println!("\n    ======================================================\n")
		}

		let inp = format!("#{i:<3} - {:<6.2} .. {:<6.2}", data[i], data[i+INPS-1]);
		let tgt = &data[i+INPS..i+INPS+2];
		let out = ai::run(agent, &data[i..i+INPS], tgt);

		let err = (out[0] - tgt[0]).abs().max((out[1] - tgt[1]).abs());

		println!("{inp} => {out:>6.2?} (err={err:<4.2}) <= {tgt:.2?}")
	}

	println!("\nNeural Network: {:#?}\n\n{:.2?}", agent.brain, agent.error);
}

pub fn progress(agent: &Agent, alive: usize, n: usize, iters: usize) {
	let maxerr = agent.error.max;
	let toterr = agent.error.tot;
	let gen    = agent.brain.gen;
	let t      = agent.runtime;

	let pb = format!("[{}>{}]", "=".repeat(n/(iters/26)), " ".repeat(26-n/(iters/26)));
	let st = format!("maxerr={maxerr:.2}, toterr={toterr:.2}, time={t:?}, gen={gen}");

	print!("\r{st:<50} {pb} (agents: {alive})");
	stdout().flush().unwrap();
}
