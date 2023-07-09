# semantic-search-rs

navie semantic search demo in Rust

## Requirements

* libtorch
* rust-bert
* hora (ANN)
* simd-json

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
  * num of docs : 16,559 documents

`hora` with `simd` feature. (need nightly build)

```text
load model : 3.0573662s
load data : 70.7628ms
inference (16559 documents) : 258.2525774s
build index : 6.4137312s
query : The story about prep school
search speed : 191µs
top 1, title : Some("Prayer for the Living")
top 2, title : Some("The Princess Diaries, Volume VI: Princess in Training")
top 3, title : Some("School Days")
top 4, title : Some("The Fall of Doctor Onslow")
top 5, title : Some("Love Lessons")
top 6, title : Some("The Turbulent Term of Tyke Tiler")
top 7, title : Some("The Fabled Fourth Graders of Aesop Elementary School")
top 8, title : Some("Flour Babies")
top 9, title : Some("The Freedom Writers Diary")
top 10, title : Some("Truancy")
```

## Dataset

* https://www.kaggle.com/datasets/ymaricar/cmu-book-summary-dataset

## Maintainer

@kozistr
