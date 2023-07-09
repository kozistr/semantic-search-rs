.PHONY: format

format:
	cargo fmt

build:
	cargo run --release --bin server

build-nightly:
	cargo +nightly run --release --bin server
