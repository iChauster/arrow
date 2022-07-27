// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <string>
#include <fstream>
#include <chrono>
#include <thread>

#include "benchmark/benchmark.h"

#include "arrow/compute/exec/test_util.h"
#include "arrow/dataset/file_parquet.h"
#include "arrow/table.h"
#include "arrow/testing/future_util.h"
#include "arrow/ipc/api.h"
#include "arrow/filesystem/api.h"
#include "arrow/util/thread_pool.h"

namespace arrow {
namespace compute {


static std::vector<std::shared_ptr<arrow::internal::ThreadPool>> internal_memory_pools;
static const char* kTimeCol = "time";
static const char* kKeyCol = "id";
const int kDefaultStart = 0;
const int kDefaultEnd = 32000;
const int kDefaultMinColumnVal = -10000;
const int kDefaultMaxColumnVal = 10000;

struct TableStats {
  std::shared_ptr<Table> table;
  size_t total_rows;
  size_t total_bytes;
};

struct ReaderNodeTableProperties {
  ExecNode* execNode;
  size_t total_rows;
  size_t total_bytes;
};

// Wrapper to enable the use of RecordBatchFileReaders as RecordBatchReaders
class RecordBatchFileReaderWrapper : public arrow::ipc::RecordBatchReader {
  std::shared_ptr<arrow::ipc::RecordBatchFileReader> _reader;
  int _next;
 public:
  virtual ~RecordBatchFileReaderWrapper() {}
  explicit RecordBatchFileReaderWrapper(
      std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader)
      : _reader(reader), _next(0) {}
  virtual arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
    // cout << "ReadNext _next=" << _next << "\n";
    if (_next < _reader->num_record_batches()) {
      ARROW_ASSIGN_OR_RAISE(*batch, _reader->ReadRecordBatch(_next++));
      // cout << "\t --> " << (*batch)->num_rows() << "\n";
    } else {
      batch->reset();
      // cout << "\t --> EOF\n";
    }
    return arrow::Status::OK();
  }
  virtual std::shared_ptr<arrow::Schema> schema() const { return _reader->schema(); }
};

static ReaderNodeTableProperties
make_arrow_ipc_reader_node(std::shared_ptr<arrow::compute::ExecPlan>& plan,
                           std::shared_ptr<arrow::fs::FileSystem>& fs,
                           const std::string& filename) {
  // TODO: error checking
  std::cout << "opening file" << std::endl;
  std::shared_ptr<arrow::io::RandomAccessFile> input = *fs->OpenInputFile(filename);
  std::cout << "file opened" << std::endl;
  std::shared_ptr<arrow::ipc::RecordBatchFileReader> in_reader =
      *arrow::ipc::RecordBatchFileReader::Open(input);
  std::shared_ptr<RecordBatchFileReaderWrapper> reader(
      new RecordBatchFileReaderWrapper(in_reader));
  auto schema = reader->schema();
  // we assume there is a time field represented in uint64, a key field of int32, and the
  // remaining fields are float64.
  size_t row_size =
      sizeof(_Float64) * (schema->num_fields() - 2) + sizeof(int64_t) + sizeof(int32_t);
  std::cout << "row_size: " << row_size << std::endl;
  auto mem_pool = arrow::internal::ThreadPool::MakeEternal(1);
  auto s = *std::move(mem_pool);
  internal_memory_pools.push_back(s);
  auto batch_gen = *arrow::compute::MakeReaderGenerator(
      std::move(reader), s.get());
  size_t rows = in_reader->CountRows().ValueOrDie();
  // cout << "create source("<<filename<<")\n";
  return {*arrow::compute::MakeExecNode(
              "source",    // registered type
              plan.get(),  // execution plan
              {},          // inputs
              arrow::compute::SourceNodeOptions(
                  std::make_shared<arrow::Schema>(*schema),  // options, )
                  batch_gen)),
          rows, row_size * rows};
}

static TableStats MakeTable(const TableGenerationProperties& properties) {
  std::shared_ptr<Table> table = MakeRandomTimeSeriesTable(properties);
  size_t row_size = sizeof(double) * (table.get()->schema()->num_fields() - 2) +
                    sizeof(int64_t) + sizeof(int32_t);
  size_t rows = table.get()->num_rows();
  return {table, rows, rows * row_size};
}

void checkMemPool(int t){
  if (internal_memory_pools.size() <= t) {
    auto mem_pool = arrow::internal::ThreadPool::MakeEternal(1);
    auto s = *std::move(mem_pool);
    internal_memory_pools.push_back(s);
  }
}

static ExecNode* MakeTableSourceNode(std::shared_ptr<arrow::compute::ExecPlan> plan,
                                     std::shared_ptr<Table> table, int batch_size, int t) {
  checkMemPool(t);
  std::shared_ptr<TableBatchReader> reader = std::make_shared<TableBatchReader>(table);
  reader->set_chunksize(batch_size);
  auto batch_gen = *arrow::compute::MakeReaderGenerator(
      std::move(reader), internal_memory_pools[t].get());
  /*return *arrow::compute::MakeExecNode(
      "table_source", plan.get(), {},
      arrow::compute::TableSourceNodeOptions(table, batch_size));*/
  return *arrow::compute::MakeExecNode(
              "source",    // registered type
              plan.get(),  // execution plan
              {},          // inputs
              arrow::compute::SourceNodeOptions(
                  table->schema(),  // options, )
                  batch_gen));
}
static void TableJoinOverhead2(benchmark::State& state,
                              int test_index,
                              std::string factory_name, ExecNodeOptions& options) {
  std::this_thread::sleep_for(std::chrono::milliseconds(5000));
  const std::string data_directory = "/home/ivan/summer/memory_benchmarking/data/";
  const std::string mem_dir = "/home/ivan/summer/arrow/cpp/memory_build/data/";

  auto file_name = mem_dir + factory_name + "_" + std::to_string(test_index);
  std::ofstream mem_files(file_name);
  std::cout << "writing to " << file_name << std::endl;
  BenchmarkMemoryPool bmp = BenchmarkMemoryPool(default_memory_pool(), mem_files);
  ExecContext ctx(&bmp, nullptr);
  std::shared_ptr<arrow::fs::FileSystem> fs =
        std::make_shared<arrow::fs::LocalFileSystem>();
  std::vector<std::pair<std::string, std::vector<std::string>>> joins = {
    {"small_small_as.feather", {"small_small_bs.feather"}},
    {"big_large_as.feather", {"big_large_bs.feather"}},
    {"big_small_as.feather", {"big_small_bs.feather"}},
    {"med_large_as.feather", {"med_large_bs.feather"}},
    {"med_small_as.feather", {"med_small_bs.feather"}},
    {"small_large_as.feather", {"small_large_bs.feather"}}
  };
  std::pair<std::string, std::vector<std::string>> v = joins[test_index];
  std::string left_table_name = data_directory + v.first;
  std::vector<std::string> right_table_names = v.second;
  int64_t rows;
  int64_t bytes;

  for (auto _ : state) {
    state.PauseTiming();
    internal_memory_pools.clear();
    ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::compute::ExecPlan> plan,
                         ExecPlan::Make(&ctx));
    auto left_table_props = make_arrow_ipc_reader_node(plan, fs, left_table_name);
    std::vector<ExecNode*> input_nodes = {
        left_table_props.execNode
    };
    rows += left_table_props.total_rows;
    bytes += left_table_props.total_bytes;
    for (std::string right_table_name : right_table_names) {
      auto right_table_props = make_arrow_ipc_reader_node(plan, fs, data_directory + right_table_name);
      input_nodes.push_back(right_table_props.execNode);
      rows += right_table_props.total_rows;
      bytes += right_table_props.total_bytes;
    }
    ASSERT_OK_AND_ASSIGN(arrow::compute::ExecNode * join_node,
                         MakeExecNode(factory_name, plan.get(), input_nodes, options));
    MakeExecNode("null_sink_node", plan.get(), {join_node}, ExecNodeOptions{});
    state.ResumeTiming();
    ASSERT_FINISHES_OK(StartAndFinish(plan.get()));
    
  }

  state.counters["total_rows_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations() *
                          rows),
      benchmark::Counter::kIsRate);

  state.counters["total_bytes_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations() *
                          bytes),
      benchmark::Counter::kIsRate);

  state.counters["maximum_peak_memory"] =
      benchmark::Counter(static_cast<double>(ctx.memory_pool()->max_memory()));
}

static void TableJoinOverhead(benchmark::State& state,
                              TableGenerationProperties left_table_properties,
                              int left_table_batch_size,
                              TableGenerationProperties right_table_properties,
                              int right_table_batch_size, int num_right_tables,
                              std::string factory_name, ExecNodeOptions& options) {
  //ExecContext ctx(default_memory_pool(), arrow::internal::GetCpuThreadPool());
  ExecContext ctx(default_memory_pool(), nullptr);

  left_table_properties.column_prefix = "lt";
  left_table_properties.seed = 0;
  TableStats left_table_stats = MakeTable(left_table_properties);

  size_t right_hand_rows = 0;
  size_t right_hand_bytes = 0;
  std::vector<TableStats> right_input_tables;
  right_input_tables.reserve(num_right_tables);

  for (int i = 0; i < num_right_tables; i++) {
    right_table_properties.column_prefix = "rt" + std::to_string(i);
    right_table_properties.seed = i + 1;
    TableStats right_table_stats = MakeTable(right_table_properties);
    right_hand_rows += right_table_stats.total_rows;
    right_hand_bytes += right_table_stats.total_bytes;
    right_input_tables.push_back(right_table_stats);
  }

  for (auto _ : state) {
    state.PauseTiming();
    int i = 0;
    ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::compute::ExecPlan> plan,
                         ExecPlan::Make(&ctx));
    std::vector<ExecNode*> input_nodes = {
        MakeTableSourceNode(plan, left_table_stats.table, left_table_batch_size, i)};
    input_nodes.reserve(right_input_tables.size() + 1);
    for (TableStats table_stats : right_input_tables) {
      ++i;
      input_nodes.push_back(
          MakeTableSourceNode(plan, table_stats.table, right_table_batch_size, i));
    }
    ASSERT_OK_AND_ASSIGN(arrow::compute::ExecNode * join_node,
                         MakeExecNode(factory_name, plan.get(), input_nodes, options));
    MakeExecNode("null_sink_node", plan.get(), {join_node}, ExecNodeOptions{});
    state.ResumeTiming();
    ASSERT_FINISHES_OK(StartAndFinish(plan.get()));
    // std::cerr << "ITERATION ENDS" << std::endl;
  }

  state.counters["total_rows_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations() *
                          (left_table_stats.total_rows + right_hand_rows)),
      benchmark::Counter::kIsRate);

  state.counters["total_bytes_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations() *
                          (left_table_stats.total_bytes + right_hand_bytes)),
      benchmark::Counter::kIsRate);

  state.counters["maximum_peak_memory"] =
      benchmark::Counter(static_cast<double>(ctx.memory_pool()->max_memory()));
}

static void AsOfJoinOverhead(benchmark::State& state) {
  int64_t tolerance = 0;
  AsofJoinNodeOptions options = AsofJoinNodeOptions(kTimeCol, kKeyCol, tolerance);
  TableJoinOverhead(
      state,
      TableGenerationProperties{int(state.range(0)), int(state.range(1)),
                                int(state.range(2)), "", kDefaultMinColumnVal,
                                kDefaultMaxColumnVal, 0, kDefaultStart, kDefaultEnd},
      int(state.range(3)),
      TableGenerationProperties{int(state.range(5)), int(state.range(6)),
                                int(state.range(7)), "", kDefaultMinColumnVal,
                                kDefaultMaxColumnVal, 0, kDefaultStart, kDefaultEnd},
      int(state.range(8)), int(state.range(4)), "asofjoin", options);
}

static void HashJoinOverhead(benchmark::State& state) {
  HashJoinNodeOptions options =
      HashJoinNodeOptions({kTimeCol, kKeyCol}, {kTimeCol, kKeyCol});
  TableJoinOverhead(
      state,
      TableGenerationProperties{int(state.range(0)), int(state.range(1)),
                                int(state.range(2)), "", kDefaultMinColumnVal,
                                kDefaultMaxColumnVal, 0, kDefaultStart, kDefaultEnd},
      int(state.range(3)),
      TableGenerationProperties{int(state.range(5)), int(state.range(6)),
                                int(state.range(7)), "", kDefaultMinColumnVal,
                                kDefaultMaxColumnVal, 0, kDefaultStart, kDefaultEnd},
      int(state.range(8)), int(state.range(4)), "hashjoin", options);
}

static void MemoryAsOfJoinOverhead(benchmark::State& state, int test_index) {
  std::cout << getpid() << std::endl;
  int64_t tolerance = 0;
  AsofJoinNodeOptions options = AsofJoinNodeOptions(kTimeCol, kKeyCol, tolerance);
  TableJoinOverhead2(state, test_index, "asofjoin", options);
}
int default_freq = 400;
int default_cols = 20;
int default_ids = 500;
int default_num_tables = 1;
int default_batch_size = 4000;
// this generates the set of right hand tables to test on.
void SetArgs(benchmark::internal::Benchmark* bench) {
  bench
      ->ArgNames({"left_freq", "left_cols", "left_ids", "left_batch_size",
                  "num_right_tables", "right_freq", "right_cols", "right_ids",
                  "right_batch_size"})
      ->UseRealTime();

  std::vector<int> v = {50, 100, 400, 1000, 2000};
  for (int freq : v){ //1, 5, 10}) {
    for (int freq2 : v) {
      bench->Args({freq, default_cols, default_ids, default_batch_size, default_num_tables,
                 freq2, default_cols, default_ids, default_batch_size});
    }
  }
  v = {10,20,100};
  for (int cols : v) {
    for (int cols2 : v) {
    bench->Args({default_freq, cols, default_ids, default_batch_size, default_num_tables,
                 default_freq, cols2, default_ids, default_batch_size});
    }
  }
  v = {10, 100, 500, 1000, 2000};
  for (int ids : v) {
    for (int ids2 : v) {
    bench->Args({default_freq, default_cols, ids, default_batch_size, default_num_tables,
                 default_freq, default_cols, ids2, default_batch_size});
    }
  }
  v = {1000, 4000, 16000};
  for (int batch_size : v) {
    for (int batch_size2 : v) {
    bench->Args({default_freq, default_cols, default_ids, batch_size, default_num_tables,
                 default_freq, default_cols, default_ids, batch_size2});
    }
  }
}

void MultiTableArgs(benchmark::internal::Benchmark* bench) {
  bench
      ->ArgNames({"left_freq", "left_cols", "left_ids", "left_batch_size",
                  "num_right_tables", "right_freq", "right_cols", "right_ids",
                  "right_batch_size"})
      ->UseRealTime();
  for (int num_tables : {1, 2, 4}) {
    bench->Args({default_freq, default_cols, default_ids, default_batch_size, num_tables,
                 default_freq, default_cols, default_ids, default_batch_size});
  }
}

BENCHMARK(AsOfJoinOverhead)->Apply(SetArgs);
BENCHMARK(AsOfJoinOverhead)->Apply(MultiTableArgs);
BENCHMARK(HashJoinOverhead)->Apply(SetArgs);
// BENCHMARK_CAPTURE(MemoryAsOfJoinOverhead, testing1, 1);
}  // namespace compute
}  // namespace arrow
