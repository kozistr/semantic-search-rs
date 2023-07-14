# semantic-search-rs

navie semantic search demo with gRPC server in Rust

## Goals

* cost-effective billion-scale vector serach in a single digit latency (< 10 ms)
* stateless searcher
  * single machine, but lots of RAM, disk space
* software & hardware-level optimizations to effectively utilize the resources

## Non-Goals

* distributed & sharded index building & searching

## To-Do

* [x] modify hnswlib-rs
* [ ] memmap the `.data` (offload to the local disk) to reduce the memory usage
* [ ] separate embedding and search part as a different micro service
* [ ] (optional) hybrid HNSW-IF indexing
* [ ] (optional) quantize & reduce the embedding dimension

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

## Benchmarks

* GPU : GTX 1060 6G (CUDA 11.8, CuDNN 8.8.x)
* CPU : i7-7700K
* Info
  * packages are compiled with `AVX2`, `FMA` flags (RUSTCFLAGS)
  * do some optimizations at compile-time
  * indexing : HNSW (FLAT)
  * embedding dimenstion : 384
  * distance measure : L2
  * k : 10
  * warm up with 10 times

### CMU Book Summary dataset

* num of documents : 16,559 documents

|  batch size | requests |   k    |  type  |    mean    |    p95     |     p99    |    p99.9   |     max    |
|    :---:    |  :---:   | :---:  | :---:  |    :---:   |    :---:   |    :---:   |    :---:   |   :---:    |
|       1     |   10k    |   10   | total  |   7.335 ms |   7.623 ms |   8.159 ms |   8.754 ms |  10.203 ms |
|             |          |        | model  |   7.067 ms |   7.279 ms |   7.734 ms |   8.312 ms |   9.710 ms |
|             |          |        | search |   0.156 ms |   0.220 ms |   0.267 ms |   0.312 ms |   0.346 ms |
|      32     |    1k    |   10   | total  |  10.119 ms |  10.795 ms |  11.263 ms |  13.839 ms |  13.839 ms |
|             |          |        | model  |   9.142 ms |   9.753 ms |  10.133 ms |  11.507 ms |  11.507 ms |
|             |          |        | search |   0.749 ms |   0.848 ms |   0.909 ms |   1.052 ms |   1.052 ms |
|     128     |    1k    |   10   | total  |  31.811 ms |  32.348 ms |  33.224 ms |  41.531 ms |  41.531 ms |
|             |          |        | model  |  29.265 ms |  29.772 ms |  30.604 ms |  31.926 ms |  31.926 ms |
|             |          |        | search |   2.338 ms |   2.495 ms |   2.615 ms |   3.166 ms |   3.166 ms |

* QPS
  * total (mean)
    * bs 1   :  136 QPS
    * bs 32  : 3162 QPS
    * bs 128 : 4024 QPS
  * search (mean)
    * bs 1   :  6410 QPS
    * bs 32  : 42724 QPS
    * bs 128 : 54748 QPS

### AG News dataset

* num of documents : 127.6K documents

|  batch size | requests |   k    |  type  |    mean    |    p95     |     p99    |    p99.9   |     max    |
|    :---:    |  :---:   | :---:  | :---:  |    :---:   |    :---:   |    :---:   |    :---:   |   :---:    |
|       1     |   10k    |   10   | total  |   4.833 ms |   5.426 ms |   6.037 ms |   9.578 ms |  12.411 ms |
|             |          |        | model  |   4.507 ms |   4.994 ms |   5.505 ms |   8.979 ms |  12.094 ms |
|             |          |        | search |   0.203 ms |   0.308 ms |   0.354 ms |   0.449 ms |   0.567 ms |
|      32     |    1k    |   10   | total  |  10.403 ms |  10.951 ms |  11.625 ms |  16.794 ms |  16.794 ms |
|             |          |        | model  |   9.211 ms |   9.684 ms |  10.226 ms |  14.796 ms |  14.796 ms |
|             |          |        | search |   0.981 ms |   1.183 ms |   1.386 ms |   2.386 ms |   2.386 ms |
|     128     |    1k    |   10   | total  |  33.440 ms |  32.602 ms |  32.862 ms |  33.440 ms |  33.440 ms |
|             |          |        | model  |  29.120 ms |  29.411 ms |  29.653 ms |  30.333 ms |  30.333 ms |
|             |          |        | search |   2.855 ms |   3.007 ms |   3.154 ms |   3.452 ms |   3.452 ms |

* QPS
  * total (mean)
    * bs 1   :  207 QPS
    * bs 32  : 3076 QPS
    * bs 128 : 3828 QPS
  * search (mean)
    * bs 1   :  4926 QPS
    * bs 32  : 32620 QPS
    * bs 128 : 44834 QPS

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

## Datasets

* https://huggingface.co/datasets/ag_news
* https://www.kaggle.com/datasets/ymaricar/cmu-book-summary-dataset

## References

* https://github.com/hora-search/hora
* https://github.com/jean-pierreBoth/hnswlib-rs

## Maintainer

@kozistr
