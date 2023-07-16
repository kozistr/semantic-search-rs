# semantic-search-rs

navie semantic search demo with gRPC server in Rust

## Goals

* cost-effective billion-scale vector serach in a single digit latency (< 10 ms)
* stateless vector searcher
* software & hardware-level optimizations to effectively utilize the resources

## Non-Goals

* distributed & sharded index building & searching
* quantize & reduce the embedding dimension
* support various indexing algorithms

### real thing does matter

To serve billion-scale vector search in real-time, effectively, there're two things to be achieved, followed by [this post](https://0x65.dev/blog/2019-12-07/indexing-billions-of-text-vectors.html).

1. vector
    * reduce vector size
    * quantize vector
2. offload the index (to the local disk)
    * to save memory

## To-Do

* [x] modify hnswlib-rs
  * [x] re-implement distance calculation (L2, Cosine) with SIMD (more effective, x2 faster)
  * [x] resolve build issue
* [ ] memmap the `.data` (offload to the local disk) to reduce the memory usage
* [ ] separate embedding and search part as a different micro service
* [ ] (optional) hybrid HNSW-IF indexing

## Architecture

### Features

* gRPC server
* dynamic batch inference (both of model & search)
* inference a LM in a real time on the GPU.
  * model info : [hf - Mini-LM L12 v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
* ANN (modified hnswlib-rs)
  * multi-threaded insertion and search.
  * SIMD accelrated distance calculation. (re-implemented)
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

### Example

Run example with the given query. (there must be a built index with `ag_news` dataset)

```shell
cargo +nightly run --release --features example --bin main "query"
```

## Benchmarks

* GPU : GTX 1060 6G (CUDA 11.8, CuDNN 8.8.x)
* CPU : i7-7700K
* Info
  * compiled with `AVX2`, `FMA` flags (RUSTCFLAGS)
  * indexing : HNSW (FLAT)
  * embedding dimenstion : 384
  * distance measure : L2, Cosine
  * k : 10
  * warm up with 10 times

### CMU Book Summary dataset

* num of documents : 16,559 documents

| dist  |  bs   |  reqs  |   k   |  type  |    mean    |    p95     |     p99    |    p99.9   |     max    |
| :---: | :---: |  :---: | :---: | :---:  |    :---:   |    :---:   |    :---:   |    :---:   |   :---:    |
|  L2   |    1  |   10k  |  10   | total  |   7.335 ms |   7.623 ms |   8.159 ms |   8.754 ms |  10.203 ms |
|       |       |        |       | model  |   7.067 ms |   7.279 ms |   7.734 ms |   8.312 ms |   9.710 ms |
|       |       |        |       | search |   0.156 ms |   0.220 ms |   0.267 ms |   0.312 ms |   0.346 ms |
|       |   32  |   1k   |  10   | total  |  10.119 ms |  10.795 ms |  11.263 ms |  13.839 ms |  13.839 ms |
|       |       |        |       | model  |   9.142 ms |   9.753 ms |  10.133 ms |  11.507 ms |  11.507 ms |
|       |       |        |       | search |   0.749 ms |   0.848 ms |   0.909 ms |   1.052 ms |   1.052 ms |
|       |  128  |   1k   |  10   | total  |  31.811 ms |  32.348 ms |  33.224 ms |  41.531 ms |  41.531 ms |
|       |       |        |       | model  |  29.265 ms |  29.772 ms |  30.604 ms |  31.926 ms |  31.926 ms |
|       |       |        |       | search |   2.338 ms |   2.495 ms |   2.615 ms |   3.166 ms |   3.166 ms |

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

| dist  |  bs   |  reqs  |   k   |  type  |    mean    |    p95     |     p99    |    p99.9   |     max    |
| :---: | :---: |  :---: | :---: | :---:  |    :---:   |    :---:   |    :---:   |    :---:   |   :---:    |
|  L2   |    1  |   10k  |  10   | total  |   4.833 ms |   5.426 ms |   6.037 ms |   9.578 ms |  12.411 ms |
|       |       |        |       | model  |   4.507 ms |   4.994 ms |   5.505 ms |   8.979 ms |  12.094 ms |
|       |       |        |       | search |   0.203 ms |   0.308 ms |   0.354 ms |   0.449 ms |   0.567 ms |
|       |   32  |   1k   |  10   | total  |  10.403 ms |  10.951 ms |  11.625 ms |  16.794 ms |  16.794 ms |
|       |       |        |       | model  |   9.211 ms |   9.684 ms |  10.226 ms |  14.796 ms |  14.796 ms |
|       |       |        |       | search |   0.981 ms |   1.183 ms |   1.386 ms |   2.386 ms |   2.386 ms |
|       |  128  |   1k   |  10   | total  |  32.220 ms |  32.602 ms |  32.862 ms |  33.440 ms |  33.440 ms |
|       |       |        |       | model  |  29.120 ms |  29.411 ms |  29.653 ms |  30.333 ms |  30.333 ms |
|       |       |        |       | search |   2.855 ms |   3.007 ms |   3.154 ms |   3.452 ms |   3.452 ms |
| Cos   |  128  |   1k   |  10   | total  |  31.547 ms |  31.890 ms |  32.087 ms |  33.479 ms |  33.479 ms |
|       |       |        |       | model  |  28.992 ms |  29.229 ms |  29.438 ms |  30.936 ms |  30.936 ms |
|       |       |        |       | search |   2.275 ms |   2.444 ms |   2.595 ms |   2.722 ms |   2.722 ms |

* QPS
  * total (mean, L2)
    * bs 1   :  207 QPS
    * bs 32  : 3076 QPS
    * bs 128 : 3973 QPS
  * search (mean, L2)
    * bs 1   :  4926 QPS
    * bs 32  : 32620 QPS
    * bs 128 : 44834 QPS
  * search (mean, Cosine)
    * bs 128 : 56264 QPS

## Examples

### CMU Book

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

### AG News

* query : `Asia shares drift lower as investors factor in Fed rate hike.`

Cosine distance

```text
top 1 | id : 92048, dist : 0.3335874
Asian Shares Lower as Oil Fall Stalls  SINGAPORE (Reuters) - Asian share markets saw light selling  in line with a slowing retreat in oil prices on Tuesday, while  the dollar steadied as officials in Europe and Japan fretted  about the U.S. currency's recent decline.
top 2 | id : 65687, dist : 0.38142937
Asian Shares Lower on Earnings, Steel  SINGAPORE (Reuters) - Asian share markets staged a  broad-based retreat on Wednesday, led by steelmakers amid  warnings of price declines, but also enveloping technology and  financial stocks on worries that earnings may disappoint.
top 3 | id : 15361, dist : 0.39869058
Asian Stocks Mostly Lower on Tech Worries  SINGAPORE (Reuters) - Asian shares were mostly lower on  Tuesday, pulled down by technology stocks following worries  about a sales forecast from bellwether Intel Corp. and  unexpectedly flat industrial output in Japan.
top 4 | id : 63234, dist : 0.40184337
Asian Stocks Slip as Oil Moves Above \$55 Asian share markets fell on Monday after oil hit a record above \$55 a barrel, with investors cautious ahead of earnings reports from bellwether firms IBM and Texas Instruments later in the day.
top 5 | id : 76590, dist : 0.4137212
Asian Stocks Ease After China Rate Rise  SINGAPORE (Reuters) - China's surprise interest rate rise  weighed down Asian stocks on Friday as investors sold shares of  miners, shippers and other firms whose fortunes have been  closely linked to the country's rapid growth.
top 6 | id : 65572, dist : 0.4192986
Asian Shares Lower on Earnings  SINGAPORE (Reuters) - Asian share markets slumped on  Wednesday, led by South Korea's POSCO Co. Ltd. and other  steelmakers amid warnings of a drop in prices, and as investors  worried that third-quarter earnings may disappoint.
top 7 | id : 14210, dist : 0.41932273
Asian Stocks Mostly Flat to Lower  SINGAPORE (Reuters) - Asian shares were mostly flat to  lower on Monday as investors kept a wary eye on stabilizing oil  prices and looked past signs Japanese consumers were beginning  to spend more freely.
top 8 | id : 14904, dist : 0.4203742
European Shares Drift Down in Thin Trade European stocks were slightly lower by mid-day in holiday-hit trading on Monday, with crude oil rises weighing on shares, though Sanofi-Aventis gained on the result of trials of an anti-obesity drug.
top 9 | id : 76632, dist : 0.4244548
Asian Stocks Ease After China Rate Rise (Reuters) Reuters - China's surprise interest rate rise\weighed down Asian stocks on Friday as investors sold shares of\miners, shippers and other firms whose fortunes have been\closely linked to the country's rapid growth.
top 10 | id : 15265, dist : 0.42584515
Asian Stocks Mostly Lower on Tech Worries (Reuters) Reuters - Asian shares were mostly lower on\Tuesday, pulled down by technology stocks following worries\about a sales forecast from bellwether Intel Corp. and\unexpectedly flat industrial output in Japan.
```

L2 distance

```text
top 1 | id : 92048, dist : 0.667175
Asian Shares Lower as Oil Fall Stalls  SINGAPORE (Reuters) - Asian share markets saw light selling  in line with a slowing retreat in oil prices on Tuesday, while  the dollar steadied as officials in Europe and Japan fretted  about the U.S. currency's recent decline.
top 2 | id : 65687, dist : 0.76285875
Asian Shares Lower on Earnings, Steel  SINGAPORE (Reuters) - Asian share markets staged a  broad-based retreat on Wednesday, led by steelmakers amid  warnings of price declines, but also enveloping technology and  financial stocks on worries that earnings may disappoint.
top 3 | id : 15361, dist : 0.7973814
Asian Stocks Mostly Lower on Tech Worries  SINGAPORE (Reuters) - Asian shares were mostly lower on  Tuesday, pulled down by technology stocks following worries  about a sales forecast from bellwether Intel Corp. and  unexpectedly flat industrial output in Japan.
top 4 | id : 63234, dist : 0.80368686
Asian Stocks Slip as Oil Moves Above \$55 Asian share markets fell on Monday after oil hit a record above \$55 a barrel, with investors cautious ahead of earnings reports from bellwether firms IBM and Texas Instruments later in the day.
top 5 | id : 76590, dist : 0.82744247
Asian Stocks Ease After China Rate Rise  SINGAPORE (Reuters) - China's surprise interest rate rise  weighed down Asian stocks on Friday as investors sold shares of  miners, shippers and other firms whose fortunes have been  closely linked to the country's rapid growth.
top 6 | id : 65572, dist : 0.83859724
Asian Shares Lower on Earnings  SINGAPORE (Reuters) - Asian share markets slumped on  Wednesday, led by South Korea's POSCO Co. Ltd. and other  steelmakers amid warnings of a drop in prices, and as investors  worried that third-quarter earnings may disappoint.
top 7 | id : 14210, dist : 0.83864534
Asian Stocks Mostly Flat to Lower  SINGAPORE (Reuters) - Asian shares were mostly flat to  lower on Monday as investors kept a wary eye on stabilizing oil  prices and looked past signs Japanese consumers were beginning  to spend more freely.
top 8 | id : 14904, dist : 0.8407483
European Shares Drift Down in Thin Trade European stocks were slightly lower by mid-day in holiday-hit trading on Monday, with crude oil rises weighing on shares, though Sanofi-Aventis gained on the result of trials of an anti-obesity drug.
top 9 | id : 77269, dist : 0.84736
Asian Markets Mixed on China Rate Hike (AP) AP - Asian financial markets showed a mixed reaction Friday to China's first interest rate rise in nine years, which analysts welcomed as a shift toward capitalist-style economic tools and away from central planning.
top 10 | id : 76632, dist : 0.8489096
Asian Stocks Ease After China Rate Rise (Reuters) Reuters - China's surprise interest rate rise\weighed down Asian stocks on Friday as investors sold shares of\miners, shippers and other firms whose fortunes have been\closely linked to the country's rapid growth.
```

## Datasets

* https://huggingface.co/datasets/ag_news
* https://www.kaggle.com/datasets/ymaricar/cmu-book-summary-dataset

## References

* https://github.com/hora-search/hora
* https://github.com/jean-pierreBoth/hnswlib-rs

## Maintainer

@kozistr
