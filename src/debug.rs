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

	let mut real_data_vec = vec![];
	let mut predicted_vec = vec![];

	let j = INPS_SIZE;
	for i in 0..real_data.len() {
		real_data_vec.push(vec![i     as f64, real_data[i][0], real_data[i][1]]);
		predicted_vec.push(vec![(i+j) as f64, predicted[i][0], predicted[i][1]]);
	}

	let chart = Chart::new()
		.legend(Legend::new().top("bottom"))
		.x_axis(
			Axis::new()
				.type_(AxisType::Category)
				.data(vec![""; real_data.len()])
				.boundary_gap(false)
				.axis_line(AxisLine::new().on_zero(false))
				.split_line(SplitLine::new().show(false))
				.min("dataMin")
				.max("dataMax")
				.axis_pointer(AxisPointer::new().z(100)),
		)
		.y_axis(
			Axis::new()
				.scale(true)
				.split_area(SplitArea::new().show(true)),
		)
		.series(
			Custom::new()
				.name("Actual")
				.dimensions(vec!["-", "lowest", "highest"])
				.encode(
					DimensionEncode::new()
						.x(0)
						.y(vec![1, 2])
						.tooltip(vec![1, 2])
				)
				.render_item(RENDER_ITEM)
				.data(real_data_vec)
		)
		.series(
			Custom::new()
				.name("Predicted")
				.dimensions(vec!["-", "lowest", "highest"])
				.encode(
					DimensionEncode::new()
						.x(0)
						.y(vec![1, 2])
						.tooltip(vec![1, 2])
				)
				.render_item(RENDER_ITEM)
				.data(predicted_vec)
		);

	HtmlRenderer::new("chart", 1000, 800).save(&chart, "predict_chart.html").unwrap();

	open::that("predict_chart.html")
}

static RENDER_ITEM: &str = r#"
function (params, api) {
	var xValue = api.value(0);
	var lowPoint = api.coord([xValue, api.value(1)]);
	var highPoint = api.coord([xValue, api.value(2)]);
	var style = api.style({
		stroke: api.visual('color')
	});
	return {
		type: 'group',
		children: [
			{
				type: 'line',
				shape: {
					x1: lowPoint[0],
					y1: lowPoint[1],
					x2: highPoint[0],
					y2: highPoint[1]
				},
				style: style
			}
		]
	};
}
"#;
