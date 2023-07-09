# semantic-search-rs

navie semantic search demo in Rust

## Requirements

* libtorch
* rust-bert
* hora (ANN)

## Run

### Server

```shell
cargo run --release --bin server 
```

```shell
cargo +nightly run --release --bin server 
```

## Performance

### Example

* GPU : GTX 1060 6G (used at extracting embedding)
* CPU : i7-7700K
* Info
  * packages are compiled with `AVX2`, `FMA` flags
  * HNSW index is used
  * do some optimization at compile-time
  * embedding dimenstion : 384
  * distance measure : L2

`hora` without `simd` feature.

```text
load data : 97.2µs
batch inference (10 documents) : 296.2011ms
set index : 128.1µs
Querying: The story about prep school
search speed : 16.6µs
top 1, title : Some("The Catcher in the Rye")
top 2, title : Some("The Great Gatsby")
top 3, title : Some("The Grapes of Wrath 4")
top 4, title : Some("The Grapes of Wrath 2")
top 5, title : Some("The Grapes of Wrath 3")
```

`hora` with `simd` feature. (need nightly build)

```text
load data : 101.8µs
batch inference (10 documents) : 281.4479ms
set index : 106.2µs
Querying: The story about prep school
search speed : 6.2µs
top 1, title : Some("The Catcher in the Rye")
top 2, title : Some("The Great Gatsby")
top 3, title : Some("The Grapes of Wrath 4")
top 4, title : Some("The Grapes of Wrath 2")
top 5, title : Some("The Grapes of Wrath 5")
```

## Reference

* https://sachaarbonel.medium.com/how-to-build-a-semantic-search-engine-in-rust-e96e6378cfd9

## Maintainer

@kozistr
