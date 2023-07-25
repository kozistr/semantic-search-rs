# semantic-search-rs

semantic search demo with gRPC server in Rust

## Goals

* cost-effective billion-scale vector serach in a single digit latency (< 10 ms)
* stateless vector searcher
* software & hardware-level optimizations to effectively utilize the resources

## Non-Goals

* distributed & sharded index building & searching
* utilze any vector database or engine products
* support various indexing algorithms
* reduce the embedding dimension

## To-Do

* [x] modify [hnswlib-rs](https://github.com/jean-pierreBoth/hnswlib-rs)
  * [x] re-implement distance calculation (L1, L2, Cosine) with SIMD (more effective, support more types, x2 faster)
  * [x] resolve build issue
  * [x] support quantization & i8 simd vector search
* [ ] memmap the `.data` (offload to the local disk) to reduce the memory usage
* [ ] separate embedding and search part as a different micro service
* [ ] (optional) hybrid HNSW-IF indexing

## Architecture

### Features

* gRPC server
* dynamic batch inference (both of model & search)
* inference a LM in a real time on the GPU.
  * model info : [hf - Mini-LM L12 v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
* vector quantization (post-quantization)
  * rescale f32 to i8
* ANN (modified hnswlib-rs)
  * multi-threaded insertion and search.
  * SIMD accelrated distance calculation. (re-implemented)
  * HNSW index

### Data

* Input  : queries. list of String.
* Output : top k indices. 2d list of int32.

## Requirements

* Intel (modern) CPU (which supports AVX2, FMA instructions)
  * some of distance measures doesn't support M1 Silcon or AMD CPUs.
* [Rust (nightly)](https://doc.rust-lang.org/book/appendix-07-nightly-rust.html)
* [libtorch 2.0](https://github.com/LaurentMazare/tch-rs#libtorch-manual-install)
* [protobuf](https://github.com/protocolbuffers/protobuf)

## Run

### Build index

Extract embeddings from the given documents and build & save an index to the local disk.

```shell
cargo +nightly run --release --features progress --bin embedding
```

If you want i8 vector (quantized vector), pass `quantize` to the argument.

```shell
cargo +nightly run --release --features progress --bin embedding quantize
```

### gRPC Server

Build & Run gRPC server (for model & search inference).

```shell
make server
```

### gRPC Client

Build & Run gRPC client. The client will start to benchmark the server based on the given parameters.

```shell
make client
```

You can also change the arguments. e.g. `./client num_users num_requests bs k`

```shell
cargo +nightly run --release --bin client 1 1000 128 10
```

### Example

Run example with the given query. (there must be a built index with `ag_news` dataset)

```shell
cargo +nightly run --release --features progress --bin main "query"
```

## Benchmarks

* GPU : GTX 1060 6G (CUDA 11.8, CuDNN 8.8.x)
* CPU : i7-7700K (4 cores 8 threads, not overclocked)
* Info
  * compiled with `AVX2`, `FMA` flags
  * indexing : HNSW
  * embedding dimenstion : 384
  * embedding data type : f32, i8
  * distance measure : L2, Cosine
  * k : 10
  * num of documents : 127.6K documents
  * warm up with 11 times

% latency is bit different by rust version. In the recent version (1.73.0 nightly), the speed becomes slower (2.275 ms -> 3.000 ms).

% i8 is benchmarked with 1.73.0.

|   p   | dist  |  bs   |  reqs  |   k   |  type  |    mean    |    p95     |     p99    |    p99.9   |     max    |
| :---: | :---: | :---: |  :---: | :---: | :---:  |    :---:   |    :---:   |    :---:   |    :---:   |   :---:    |
|  f32  |  L2   |    1  |   10k  |  10   | total  |   4.833 ms |   5.426 ms |   6.037 ms |   9.578 ms |  12.411 ms |
|       |       |       |        |       | model  |   4.507 ms |   4.994 ms |   5.505 ms |   8.979 ms |  12.094 ms |
|       |       |       |        |       | search |   0.203 ms |   0.308 ms |   0.354 ms |   0.449 ms |   0.567 ms |
|       |       |   32  |   1k   |  10   | total  |  10.403 ms |  10.951 ms |  11.625 ms |  16.794 ms |  16.794 ms |
|       |       |       |        |       | model  |   9.211 ms |   9.684 ms |  10.226 ms |  14.796 ms |  14.796 ms |
|       |       |       |        |       | search |   0.981 ms |   1.183 ms |   1.386 ms |   2.386 ms |   2.386 ms |
|       |       |  128  |   1k   |  10   | total  |  32.220 ms |  32.602 ms |  32.862 ms |  33.440 ms |  33.440 ms |
|       |       |       |        |       | model  |  29.120 ms |  29.411 ms |  29.653 ms |  30.333 ms |  30.333 ms |
|       |       |       |        |       | search |   2.855 ms |   3.007 ms |   3.154 ms |   3.452 ms |   3.452 ms |
|       |  Cos  |  128  |   1k   |  10   | total  |  31.547 ms |  31.890 ms |  32.087 ms |  33.479 ms |  33.479 ms |
|       |       |       |        |       | model  |  28.992 ms |  29.229 ms |  29.438 ms |  30.936 ms |  30.936 ms |
|       |       |       |        |       | search |   2.275 ms |   2.444 ms |   2.595 ms |   2.722 ms |   2.722 ms |
|   i8  |  Cos  |  128  |   1k   |  10   | total  |  31.006 ms |  31.378 ms |  31.665 ms |  32.360 ms |  32.360 ms |
|       |       |       |        |       | model  |  28.880 ms |  29.171 ms |  29.406 ms |  30.182 ms |  30.182 ms |
|       |       |       |        |       | search |   1.785 ms |   1.944 ms |   2.015 ms |   2.193 ms |   2.193 ms |

* QPS
  * total (mean, L2)
    * bs 1   :  207 QPS
    * bs 32  : 3076 QPS
    * bs 128 : 3973 QPS
  * total (mean, Cosine, i8)
    * bs 128 : 4128 QPS
  * search (mean, L2)
    * bs 1   :  4926 QPS
    * bs 32  : 32620 QPS
    * bs 128 : 44834 QPS
  * search (mean, Cosine)
    * bs 128 : 56264 QPS
  * search (mean, Cosine, i8)
    * bs 128 : 71709 QPS

### Search Latency

* testing with large batch size (>= 1024)
* quantized vector (i8) is about 20% faster and saving about 4x times memory than f32 vector

|   p   | dist  |  bs   |  reqs  |   k   |    mean    |    p50     |    p95     |     p99    |    p99.9   |    max     |   QPS   |
| :---: | :---: | :---: |  :---: | :---: |    :---:   |    :---:   |    :---:   |    :---:   |    :---:   |   :---:    |  :---:  |
|  f32  |  cos  | 1024  |   2k   |  10   |  10.969 ms |  10.856 ms |  12.078 ms |  12.384 ms |  14.817 ms |  18.024 ms |   93354 |
|       |       | 2048  |   2k   |  10   |  21.829 ms |  21.617 ms |  23.952 ms |  24.514 ms |  25.892 ms |  26.171 ms |   93820 |
|       |       | 4096  |   2k   |  10   |  43.538 ms |  43.199 ms |  46.303 ms |  48.374 ms |  50.194 ms |  57.159 ms |   94079 |
|       |       | 8192  |   2k   |  10   |  86.234 ms |  85.684 ms |  90.645 ms |  93.850 ms | 101.188 ms | 102.607 ms |   94997 |
|   i8  |  cos  | 1024  |   2k   |  10   |   9.170 ms |   9.082 ms |  10.153 ms |  10.317 ms |  11.047 ms |  11.188 ms |  111668 |
|       |       | 2048  |   2k   |  10   |  18.241 ms |  18.057 ms |  19.892 ms |  20.558 ms |  23.500 ms |  29.330 ms |  112275 |
|       |       | 4096  |   2k   |  10   |  36.469 ms |  36.014 ms |  40.318 ms |  41.312 ms |  42.603 ms |  43.144 ms |  112315 |
|       |       | 8192  |   2k   |  10   |  72.644 ms |  71.954 ms |  77.323 ms |  79.242 ms |  86.294 ms |  87.690 ms |  112769 |

* Macbook Pro 16-inch 2019, i9 8 cores 16 threads (2.3 GHz)

|   p   | dist  |  bs   |  reqs  |   k   |    mean    |    p50     |    p95     |     p99    |    p99.9   |    max     |   QPS   |
| :---: | :---: | :---: |  :---: | :---: |    :---:   |    :---:   |    :---:   |    :---:   |    :---:   |   :---:    |  :---:  |
|   i8  |  cos  | 1024  |   2k   |  10   |   7.003 ms |   6.700 ms |   8.270 ms |   8.668 ms |   9.655 ms |  10.011 ms |  146223 |
|       |       | 2048  |   2k   |  10   |  14.388 ms |  14.419 ms |  15.637 ms |  16.067 ms |  19.237 ms |  20.231 ms |  142341 |
|       |       | 4096  |   2k   |  10   |  27.894 ms |  27.815 ms |  29.818 ms |  31.213 ms |  39.733 ms |  40.387 ms |  146842 |
|       |       | 8192  |   2k   |  10   |  54.674 ms |  54.348 ms |  57.844 ms |  61.486 ms | 100.327 ms | 135.911 ms |  149834 |

## Examples

* dataset : [ag_news](https://huggingface.co/datasets/ag_news)
* query   : `Asia shares drift lower as investors factor in Fed rate hike.`

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

Cosine distance w/ i8 (quantized)

```text
top 1 | id : 70945, dist : 8807
Asian Shares Fall on Dollar; Gold, Oil Up (Reuters) Reuters - The U.S. dollar stumbled on Monday,\setting multi-month lows against the yen and euro and trading\around four-year lows against the Korean won and Singapore\dollar, prompting investors to clip Asian share markets.
top 2 | id : 33140, dist : 10982
US Stock-Index Futures Decline; Citigroup, GE Slip in Europe US stock-index futures declined. Dow Jones Industrial Average shares including General Electric Co. slipped in Europe. Citigroup Inc.
top 3 | id : 33077, dist : 11208
US Stock-Index Futures Are Little Changed; Citigroup, GE Slip US stock-index futures were little changed. Dow Jones Industrial Average shares including General Electric Co. slipped in Europe. Citigroup Inc.
top 4 | id : 95897, dist : 11290
Red hot Google shares cooling  Google Inc. may be subject to the law of gravity, after all. Its shares, which more than doubled in the two months after the company's Aug. 19 initial public offering and traded as high as \$201.60 on Nov. 3, have slipped about 17 percent over the past two weeks. They closed at \$167.54 yesterday, down 2.88 percent for the day, ...
top 5 | id : 40354, dist : 11390
Tokyo stocks open slightly lower TOKYO - Stocks opened slightly lower Monday on the Tokyo Stock Exchange as declines in US technology shares last Friday prompted selling.
top 6 | id : 64025, dist : 12241
E*Trade Profit Rises as Expenses Drop (Reuters) Reuters - E*Trade Financial Corp. , an\online bank and brokerage, on Monday said its third-quarter\profit rose as lower expenses offset a drop in net revenue.
top 7 | id : 58035, dist : 12458
UK inflation rate fall continues The UK's inflation rate fell in September, thanks in part to a fall in the price of airfares, increasing the chance that interest rates will be kept on hold.
top 8 | id : 65573, dist : 12583
Yen Eases Against Dollar  TOKYO (Reuters) - The yen edged down against the dollar on  Wednesday as Japanese stock prices slid, but it remained within  striking distance of three-month highs on persistent concerns  about the health of the U.S. economy.
top 9 | id : 50470, dist : 12828
Tower Auto Sees Wider Loss, Shares Fall  CHICAGO (Reuters) - Auto parts maker Tower Automotive Inc.  &lt;A HREF="http://www.investor.reuters.com/FullQuote.aspx?ticker=TWR.N target=/stocks/quickinfo/fullquote"&gt;TWR.N&lt;/A&gt; said on Tuesday its third-quarter loss would be twice  as deep as previously expected because of lower vehicle  production in North America and higher steel costs.
top 10 | id : 1267, dist : 13020
Study: Global Warming Could Change Calif. (AP) AP - Global warming could cause dramatically hotter summers and a depleted snow pack in California, leading to a sharp increase in heat-related deaths and jeopardizing the water supply, according to a study released Monday.
```

## References

* <https://github.com/hora-search/hora>
* <https://github.com/jean-pierreBoth/hnswlib-rs>

## Maintainer

@kozistr
