.PHONY: format client server builder

format:
	cargo fmt

client:
	cargo +nightly run --release --bin client 1 5000 128 10

server:
	cargo +nightly run --release --bin server

builder:
	cargo +nightly run --release --features embedding --bin embedding news
