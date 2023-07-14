# semantic-search-rs

navie semantic search demo with gRPC server in Rust

## Goals

* cost-effective billion-scale vector serach in a single digit latency (< 10 ms)
* stateless searcher
  * single machine, but lots of RAM, disk space (expected about ~ 300GB)
* software & hardware-level optimizations to effectively utilize the resources

## Non-Goals

* distributed & sharded index building & searching
* quantize & reduce the embedding dimension

## To-Do

* [x] modify hnswlib-rs
* [ ] memmap the `.data` (offload to the local disk) to reduce the memory usage
* [ ] separate embedding and search part as a different micro service
* [ ] (optional) hybrid HNSW-IF indexing

## Architecture

### Features

* gRPC server
* dynamic batch
  * batch inference
  * multi-threaded search
* inference a LM in a real time on the GPU.
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
* (modified) hnswlib-rs
* protobuf

## Run

### Build index

Extract embeddings from the given documents and build & save an index to the local disk.

```shell
make run-builder
```

### gRPC Client

Run gRPC client.

```shell
make run-client
```

### gRPC Server

Run gRPC server (for model & search inference).

```shell
make run-server
```

## Performance

* GPU : GTX 1060 6G (CUDA 11.8, CuDNN 8.8.x)
* CPU : i7-7700K
* Info
  * packages are compiled with `AVX2`, `FMA` flags (RUSTCFLAGS)
  * do some optimizations at compile-time
  * indexing : HNSW (FLAT)
  * embedding dimenstion : 384
  * distance measure : L2
  * num of documents : 16,559 documents
  * k : 10

### Benchmark

* warm up with 10 times

|  batch size | requests |   k    |  type  |    mean    |    p95     |     p99    |    p99.9   |     max    |
|    :---:    |  :---:   | :---:  | :---:  |    :---:   |    :---:   |    :---:   |    :---:   |   :---:    |
|       1     |   10k    |   10   | total  |   7.335 ms |   7.623 ms |   8.159 ms |   8.754 ms |  10.203 ms |
|             |          |        | model  |   7.067 ms |   7.279 ms |   7.734 ms |   8.312 ms |   9.710 ms |
|             |          |        | search |   0.156 ms |   0.220 ms |   0.267 ms |   0.312 ms |   0.346 ms |
|      32     |    1k    |   10   | total  |  27.998 ms |  28.428 ms |  28.907 ms |  29.272 ms |  29.272 ms |
|             |          |        | model  |  27.033 ms |  27.424 ms |  27.897 ms |  28.184 ms |  28.184 ms |
|             |          |        | search |   0.749 ms |   0.848 ms |   0.909 ms |   1.052 ms |   1.052 ms |
|      64     |    1k    |   10   | total  |  51.748 ms |  55.995 ms |  61.806 ms |  79.740 ms |  79.740 ms |
|             |          |        | model  |  50.162 ms |  54.232 ms |  59.505 ms |  77.888 ms |  77.888 ms |
|             |          |        | search |   1.346 ms |   1.565 ms |   1.972 ms |   2.431 ms |   2.431 ms |
|     128     |    1k    |   10   | total  | 101.421 ms | 108.787 ms | 109.118 ms | 109.672 ms | 109.672 ms |
|             |          |        | model  |  98.811 ms | 106.137 ms | 106.458 ms | 106.813 ms | 106.813 ms |
|             |          |        | search |   2.338 ms |   2.495 ms |   2.615 ms |   3.166 ms |   3.166 ms |

* QPS
  * total (mean)
    * bs 1   :  136 QPS
    * bs 32  : 1143 QPS
    * bs 64  : 1237 QPS
    * bs 128 : 1262 QPS
  * search (mean)
    * bs 1   :  6410 QPS
    * bs 32  : 42724 QPS
    * bs 64  : 47548 QPS
    * bs 128 : 54748 QPS

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
