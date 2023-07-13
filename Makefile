.PHONY: format

format:
	cargo fmt

run-client:
	cargo +nightly run --release --bin client --features all 1 10000 10

run-server:
	cargo +nightly run --release --bin server --features all

run-builder:
	cargo +nightly run --release --bin embedding --features all
