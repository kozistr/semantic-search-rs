.PHONY: format

format:
	cargo fmt

run-client:
	cargo +nightly run --release --bin client 1 10000 1 10

run-server:
	cargo +nightly run --release --bin server

run-builder:
	cargo +nightly run --release --features embedding --bin embedding news
