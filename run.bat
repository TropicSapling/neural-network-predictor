IF [%1]==[] (
	cargo run --release < data/io.csv
) ELSE (
	cargo run --release < "%1"
)
