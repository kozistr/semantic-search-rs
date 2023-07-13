# semantic-search-rs

navie semantic search demo with gRPC server in Rust

## Goal

* cost-effective billion-scale vector serach in a single digit latency (< 10 ms)
* stateless searcher
  * single machine, but lots of RAM, disk space (expected about ~ 300GB)

## Non-Goal

* distributed & sharded index build / search

## To-Do

* [*] hnswlib-rs
* [ ] separate embedding and search part as a different micro service
* [ ] faiss-rs with GPU
* [ ] hybrid HNSW-IF indexing

## Architecture

### Features

* gRPC server
* inference a LM in real time on the GPU.
  * model info : [hf - Mini-LM L12 v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
* ANN (modified hnswlib-rs)
  * multi-threaded insertion and search.
  * SIMD accelrated distance calculation.
  * HNSW (FLAT) index

### Data

* Input  : query. String.
* Output : top k indices. list of int32.

## Requirements

* libtorch 2.0 (cuda)
* rust-bert
* ~~hora (ANN)~~
* (modified) hnswlib-rs
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

|  batch size | requests |   k    |  type  |   mean   |   p95    |   p99    |   p99.9  |    max    |
|    :---:    |  :---:   | :---:  | :---:  |  :---:   |  :---:   |   :---:  |   :---:  |   :---:   |
|       1     |   10k    |   10   | total  | 7.335 ms | 7.623 ms | 8.159 ms | 8.754 ms | 10.203 ms |
|             |          |        | model  | 7.067 ms | 7.279 ms | 7.734 ms | 8.312 ms | 9.710 ms  |
|             |          |        | search | 0.151 ms | 0.220 ms | 0.267 ms | 0.312 ms | 0.346 ms  |

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
