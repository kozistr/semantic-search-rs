# semantic-search-rs

navie semantic search demo with gRPC server in Rust

## Requirements

* libtorch
* rust-bert
* hora (ANN)

## Run

### Server

```shell
cargo run --release --bin search-server 
```

## Performance

### Example

* GPU : GTX 1060 6G (used at extracting embedding)
* CPU : i7700K
* Info
  * packages are compiled with `AVX2`, `FMA` flags (but not for `hora`)
  * HNSW index is used
  * do some optimization at compile-time
  * embedding dimenstion : 384
  * distance measure : L2

```text
batch inference (10 documents) : 283.055ms
set index : 123.1µs
Querying: The story about prep school
search speed : 16.8µs
top 1, title : Some("The Catcher in the Rye")
top 2, title : Some("The Great Gatsby")
top 3, title : Some("The Grapes of Wrath 4")
top 4, title : Some("The Grapes of Wrath 2")
top 5, title : Some("The Grapes of Wrath 3")
```

## Reference

* https://sachaarbonel.medium.com/how-to-build-a-semantic-search-engine-in-rust-e96e6378cfd9

## Maintainer

@kozistr
