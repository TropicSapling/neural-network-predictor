use std::io::{stdout, Write};
use crate::{agent::Agent, ai, DATA_SIZE, INPS};

pub fn result(agent: &mut Agent, data: [f64; DATA_SIZE]) {
	agent.toterr = 0.0;
	agent.maxerr = 0.0;
	for i in 0..(DATA_SIZE - INPS) {
		if i == DATA_SIZE - INPS*2 {
			println!("\n    ======================================================\n")
		}

		let inp = format!("#{i:<3} - {:<6.2} .. {:<6.2}", data[i], data[i+INPS-1]);
		let tgt = data[i+INPS];
		let out = ai::run(agent, &data[i..i+INPS], tgt);

		let res = out[1] - out[0];
		let err = (res - tgt).abs();

		println!("{inp} => {out:>6.2?} => {res:<6.2} (err={err:<4.2}) <= {tgt:.2}")
	}

	let (brain, maxerr, toterr) = (&agent.brain, agent.maxerr, agent.toterr);

	println!("\nNeural Network: {brain:#?}\n\nmaxerr={maxerr}\ntoterr={toterr}\n");
}

pub fn progress(agent: &Agent, n: usize, iters: usize) {
	let (maxerr, toterr) = (agent.maxerr, agent.toterr);
	let gen              = agent.brain.gen;
	let t                = agent.runtime;

	let pb = format!("[{}>{}]", "=".repeat(n/(iters/26)), " ".repeat(26-n/(iters/26)));
	let st = format!("maxerr={maxerr:.2}, toterr={toterr:.2}, time={t:?}, gen={gen}");

	print!("\r{st:<52} {pb}");
	stdout().flush().unwrap();
}
