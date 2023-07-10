# semantic-search-rs

navie semantic search demo in Rust

## Requirements

* libtorch 2.0 (cuda)
* rust-bert
* hora (ANN)
* minmalloc

## Run

### Build index

```shell
cargo +nightly run --release --bin embedding
or
make run-builder
```

### Client

```shell
cargo +nightly run --release --bin client
or
make run-client
```

### Server

```shell
cargo +nightly run --release --bin server
or
make run-server
```

## Performance

* GPU : GTX 1060 6G (CUDA 11.8, CuDNN 8.8.x)
* CPU : i7-7700K
* Info
  * packages are compiled with `AVX2`, `FMA` flags (RUSTCFLAGS)
  * do some optimizations at compile-time
  * indexing : HNSW
  * embedding dimenstion : 384
  * distance measure : L2
  * num of documents : 16,559 documents
  * k : 10

```text
load model : 3.1107782s
load data : 71.7013ms
[-] there's no index file.
inference (16559 documents) : 91.7753841s
build index : 6.4097303s
query : The story about prep school
search speed : 288.5µs
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

```text
load model : 3.0460236s
load data : 70.939ms
[+] there's an index file.
query : The story about prep school
search speed : 255.8µs
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
