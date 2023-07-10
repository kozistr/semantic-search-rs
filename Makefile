.PHONY: format

format:
	cargo fmt

run-client:
	cargo +nightly run --release --bin client

run-server:
	cargo +nightly run --release --bin server

run-builder:
	cargo +nightly run --release --bin embedding
