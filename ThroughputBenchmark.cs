using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace TIMShield
{
    public class ThroughputBenchmark
    {
        private readonly TimEngine _engine;
        private readonly List<string> _dataset;
        private readonly Random _random = new Random(42);

        public ThroughputBenchmark(TimEngine engine, List<string> dataset)
        {
            _engine = engine;
            _dataset = dataset;
        }

        public void Run(int totalQueries = 5000, int topK = 5, int? maxDegreeOfParallelism = null)
        {
            int concurrency = maxDegreeOfParallelism ?? Environment.ProcessorCount;
            Console.WriteLine("\n=== Starting QPS & Throughput Benchmark ===");
            Console.WriteLine($"Total Queries:     {totalQueries:N0}");
            Console.WriteLine($"Concurrency Level: {concurrency} worker threads");
            Console.WriteLine($"Dataset Size:      {_dataset.Count:N0} records\n");

            // 1. Prepare query batch (mix of original items and mutated inputs)
            var queries = PrepareQueryBatch(totalQueries);

            // 2. Warm up JIT and thread pool
            Console.WriteLine("Warming up JIT and memory cache...");
            for (int i = 0; i < Math.Min(50, queries.Count); i++)
            {
                _engine.Search(queries[i], topK);
            }

            // 3. Concurrent execution
            var latenciesUs = new ConcurrentBag<long>();
            var parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = concurrency
            };

            Console.WriteLine("Executing concurrent query workload...");
            var overallSw = Stopwatch.StartNew();

            Parallel.ForEach(queries, parallelOptions, query =>
            {
                long startTick = Stopwatch.GetTimestamp();
                var results = _engine.Search(query, topK);
                long elapsedTicks = Stopwatch.GetTimestamp() - startTick;

                long microSeconds = (long)(elapsedTicks * (1_000_000.0 / Stopwatch.Frequency));
                latenciesUs.Add(microSeconds);
            });

            overallSw.Stop();

            // 4. Calculate metrics
            double totalSeconds = overallSw.Elapsed.TotalSeconds;
            double qps = totalQueries / totalSeconds;

            var sortedLatencies = latenciesUs.OrderBy(x => x).ToList();
            double avgLatencyMs = sortedLatencies.Average() / 1000.0;
            double p50Ms = sortedLatencies[(int)(sortedLatencies.Count * 0.50)] / 1000.0;
            double p95Ms = sortedLatencies[(int)(sortedLatencies.Count * 0.95)] / 1000.0;
            double p99Ms = sortedLatencies[(int)(sortedLatencies.Count * 0.99)] / 1000.0;

            Console.WriteLine("\n=== Throughput & Latency Results ===");
            Console.WriteLine($"Total Elapsed Time: {totalSeconds:F3} seconds");
            Console.WriteLine($"Throughput (QPS):   {qps:N1} queries/sec");
            Console.WriteLine($"Avg Latency:        {avgLatencyMs:F3} ms");
            Console.WriteLine($"p50 Latency:        {p50Ms:F3} ms");
            Console.WriteLine($"p95 Latency:        {p95Ms:F3} ms");
            Console.WriteLine($"p99 Latency:        {p99Ms:F3} ms");
            Console.WriteLine("=====================================\n");
        }

        private List<string> PrepareQueryBatch(int count)
        {
            var batch = new List<string>(count);
            for (int i = 0; i < count; i++)
            {
                string baseDomain = _dataset[_random.Next(_dataset.Count)];

                // 50% unmodified queries, 50% mutated typo queries
                if (_random.NextDouble() > 0.5 && baseDomain.Length > 4)
                {
                    int idx = _random.Next(baseDomain.Length);
                    char subChar = DataLoader.DomainAlphabet[_random.Next(DataLoader.DomainAlphabet.Length)];
                    var arr = baseDomain.ToCharArray();
                    arr[idx] = subChar;
                    batch.Add(new string(arr));
                }
                else
                {
                    batch.Add(baseDomain);
                }
            }
            return batch;
        }
    }
}