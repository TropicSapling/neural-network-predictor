use std::io::{stdout, Write, Result};
use crate::{ai, ai::*, agent::*, consts::*, data::*};

pub fn result(agent: &mut Agent, data: Data) -> Result<()> {
	let mut predictions = [[0.0; OUTS]; DATA_SIZE];

	// Run through entire data set, incl. previously held-out test set
	let mut error = Error::new();
	for i in 0..data.len()-INPS_SIZE {
		// Print a separation line once we reach the held-out test set
		if i == data.len() - SPAN_SIZE {
			print!("\n                   ");
			println!("===========================================================\n");
		}

		// Run agent
		let tgt = data[i+INPS_SIZE];
		let out = ai::run(agent, &data[i..i+INPS_SIZE], tgt, false);

		// Save output
		predictions[i] = out;

		// Calculate error
		let err = (out[0] - tgt[0]).abs() + (out[1] - tgt[1]).abs();

		// Record average & maximum errors
		error.avg += err;
		if err > error.max {
			error.max = err
		}

		// Format input
		let j   = i + INPS_SIZE - 1;
		let inp = format!("{i:0>3}..{j:0>3} - {:>6.2?}..{:>6.2?}", data[i], data[j]);

		// Print results for this input
		println!("{inp} => {out:>6.2?} (err={err:0>5.2}) <= {tgt:>6.2?}")
	}
	error.avg /= (data.len() - INPS_SIZE) as f64;

	// Print final neural network information
	println!("\nNeural Network: {:#?}\n\n{error:.2?}", agent.brain);

	// Plot predictions vs. real data
	plot(data, predictions)
}

pub fn progress(agent: &mut Agent, data: Data, alive: usize, n: usize, iters: usize) {
	// Collect errors for training & validation data sets
	let mut error = Error::new();
	for i in 0..data.len()-SPAN_SIZE {
		// Run agent
		let tgt = data[i+INPS_SIZE];
		let out = ai::run(agent, &data[i..i+INPS_SIZE], tgt, false);

		// Calculate error
		let err = (out[0] - tgt[0]).abs() + (out[1] - tgt[1]).abs();

		// Record average & maximum errors
		error.avg += err;
		if err > error.max {
			error.max = err
		}
	}
	error.avg /= (data.len() - INPS_SIZE) as f64;

	// Prepare for formatting
	let maxerr = error.max;
	let avgerr = error.avg;
	let gen    = agent.brain.gen;
	let t      = agent.runtime;

	// Format errors and other information
	let pb = format!("[{}>{}]", "=".repeat(n/(iters/26))," ".repeat(26-n/(iters/26)));
	let st = format!("maxerr={maxerr:.2}, avgerr={avgerr:.2}, time={t:?}, gen={gen}");

	// Print progress bar
	print!("\r{st:<50} {pb} (agents: {alive})");
	stdout().flush().unwrap();
}

////////////////////////////////////////////////////////////////

fn plot(real_data: Data, predicted: Data) -> Result<()> {
	use charming::{component::*, element::*, series::*, Chart, HtmlRenderer};
	use open;

	let chart = Chart::new()
		.legend(Legend::new().top("bottom"))
		.series(
			Pie::new()
				.name("Nightingale Chart")
				.rose_type(PieRoseType::Radius)
				.radius(vec!["50", "250"])
				.center(vec!["50%", "50%"])
				.item_style(ItemStyle::new().border_radius(8))
				.data(vec![
					(40.0, "rose 1"),
					(38.0, "rose 2"),
					(32.0, "rose 3"),
					(30.0, "rose 4"),
					(28.0, "rose 5"),
					(26.0, "rose 6"),
					(22.0, "rose 7"),
					(18.0, "rose 8"),
				]),
		);

	HtmlRenderer::new("chart", 1000, 800).save(&chart, "predict_chart.html").unwrap();

	open::that("predict_chart.html")
}
