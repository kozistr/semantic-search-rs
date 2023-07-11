# semantic-search-rs

navie semantic search demo with gRPC server in Rust

## Goal

* cost-effective billion-scale vector serach in a single digit latency (< 10 ms)
  * single machine, single thread
* stateless searcher

## Non-Goal

* distributed index build / search
* CRUD for index
* code-level optimization for indexing algorithm
* reduce num of embedding dimension

## To-Do

* [ ] faiss-rs with GPU
* [ ] hybrid HNSW-IF indexing

## Architecture

TBD

## Requirements

* libtorch 2.0 (cuda)
* rust-bert
* hora (ANN)
* minmalloc
* protobuf

## Run

### Build index

```shell
make run-builder
```

### gRPC Client

```shell
make run-client
```

### gRPC Server

```shell
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

### Benchmark

* warm up with 10 times

| batch size | requests |   k    |  type  |   mean   |   p95    |   p99    |   p99.9  |    max    |
|   :---:    |  :---:   | :---:  | :---:  |  :---:   |  :---:   |   :---:  |   :---:  |   :---:   |
| 1          |   10k    |        | model  | 7.240 ms | 7.485 ms | 7.830 ms | 9.250 ms | 12.263 ms |
|            |          |   10   | search | 0.141 ms | 0.209 ms | 0.247 ms | 0.310 ms | 0.376 ms  |

### Example

```text
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

## Dataset

* https://www.kaggle.com/datasets/ymaricar/cmu-book-summary-dataset

## Maintainer

@kozistr
