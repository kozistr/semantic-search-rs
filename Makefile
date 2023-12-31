.PHONY: format client server builder lint

update:
	cargo upgrade

format:
	cargo +nightly fmt

client:
	cargo +nightly run --release --bin client 1 1000 128 10

server:
	cargo +nightly run --release --bin server

builder:
	cargo +nightly run --release --features progress --bin embedding quantize

example:
	cargo +nightly run --release --bin main "Asia shares drift lower as investors factor in Fed rate hike." asdf

lint:
	cargo +nightly clippy
