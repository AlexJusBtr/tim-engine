using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace TIMShield
{
    public class RecallBenchmark
    {
        private readonly List<string> _dataset;
        private readonly TimEngine _timEngine;
        private readonly Random _random = new Random(1337);

        public RecallBenchmark(List<string> dataset, TimEngine engine)
        {
            _dataset = dataset;
            _timEngine = engine;
        }

        public void Run(int numTestQueries = 100, int topK = 5)
        {
            Console.WriteLine($"\n=== Starting Recall & Latency Benchmark ===");
            Console.WriteLine($"Dataset Size: {_dataset.Count:N0} records");
            Console.WriteLine($"Test Queries: {numTestQueries}");
            Console.WriteLine($"Top-K: {topK}\n");

            // 1. Generate test suite of mutated strings with known original sources
            var testCases = GenerateMutatedQueries(numTestQueries);

            int timFoundOriginalCount = 0;
            int totalOverlapCount = 0;
            long totalLevenshteinTimeMs = 0;
            long totalTimTimeMs = 0;

            for (int i = 0; i < testCases.Count; i++)
            {
                var (original, mutated, editType) = testCases[i];

                // --- Ground Truth: Linear Levenshtein Brute Force ---
                var levSw = Stopwatch.StartNew();
                var levResults = _dataset
                    .Select((item, idx) => (Item: item, Distance: ComputeLevenshtein(mutated, item)))
                    .OrderBy(x => x.Distance)
                    .Take(topK)
                    .Select(x => x.Item)
                    .ToHashSet();
                levSw.Stop();
                totalLevenshteinTimeMs += levSw.ElapsedMilliseconds;

                // --- Evaluated Engine: TIM Inverted Index + Cosine Search ---
                var timSw = Stopwatch.StartNew();
                var timResults = _timEngine.Search(mutated, topK)
                    .Select(r => r.MatchedItem)
                    .ToList();
                timSw.Stop();
                totalTimTimeMs += timSw.ElapsedMilliseconds;

                // Check if TIM retrieved the original unmutated target (Recall@K for source)
                if (timResults.Contains(original))
                {
                    timFoundOriginalCount++;
                }

                // Check overlap between TIM's top-K and Levenshtein's top-K
                int overlap = timResults.Count(item => levResults.Contains(item));
                totalOverlapCount += overlap;
            }

            // --- Metrics Calculation ---
            double sourceRecallAtK = (double)timFoundOriginalCount / numTestQueries * 100.0;
            double levOverlapRecallAtK = (double)totalOverlapCount / (numTestQueries * topK) * 100.0;
            double avgLevLatency = (double)totalLevenshteinTimeMs / numTestQueries;
            double avgTimLatency = (double)totalTimTimeMs / numTestQueries;

            Console.WriteLine("=== Benchmark Results ===");
            Console.WriteLine($"Source Recovery (Recall@{topK} for Original): {sourceRecallAtK:F2}%");
            Console.WriteLine($"Levenshtein Top-{topK} Overlap:              {levOverlapRecallAtK:F2}%");
            Console.WriteLine($"Avg TIM Latency:                             {avgTimLatency:F2} ms / query");
            Console.WriteLine($"Avg Linear Levenshtein Latency:              {avgLevLatency:F2} ms / query");
            Console.WriteLine($"Speedup vs Linear Levenshtein:               {(avgLevLatency / Math.Max(0.01, avgTimLatency)):F1}x\n");
        }

        private List<(string Original, string Mutated, string MutationType)> GenerateMutatedQueries(int count)
        {
            var testCases = new List<(string, string, string)>(count);
            var candidates = _dataset.Where(s => s.Length >= 5).Take(count * 2).ToList();

            for (int i = 0; i < count; i++)
            {
                string original = candidates[_random.Next(candidates.Count)];
                int mutationChoice = _random.Next(4);
                string mutated;
                string type;

                switch (mutationChoice)
                {
                    case 0: // 1-character deletion
                        int delIdx = _random.Next(original.Length);
                        mutated = original.Remove(delIdx, 1);
                        type = "Deletion";
                        break;
                    case 1: // 1-character substitution (typosquatting)
                        int subIdx = _random.Next(original.Length);
                        char subChar = DataLoader.DomainAlphabet[_random.Next(DataLoader.DomainAlphabet.Length)];
                        var charArr = original.ToCharArray();
                        charArr[subIdx] = subChar;
                        mutated = new string(charArr);
                        type = "Substitution";
                        break;
                    case 2: // 1-character insertion
                        int insIdx = _random.Next(original.Length + 1);
                        char insChar = DataLoader.DomainAlphabet[_random.Next(DataLoader.DomainAlphabet.Length)];
                        mutated = original.Insert(insIdx, insChar.ToString());
                        type = "Insertion";
                        break;
                    default: // Transposition (swapping adjacent characters)
                        int swapIdx = _random.Next(original.Length - 1);
                        var swapArr = original.ToCharArray();
                        (swapArr[swapIdx], swapArr[swapIdx + 1]) = (swapArr[swapIdx + 1], swapArr[swapIdx]);
                        mutated = new string(swapArr);
                        type = "Transposition";
                        break;
                }

                testCases.Add((original, mutated, type));
            }

            return testCases;
        }

        /// <summary>
        /// Optimized two-row dynamic programming Levenshtein distance algorithm (O(N) memory).
        /// </summary>
        public static int ComputeLevenshtein(string s, string t)
        {
            if (string.IsNullOrEmpty(s)) return t?.Length ?? 0;
            if (string.IsNullOrEmpty(t)) return s.Length;

            int n = s.Length;
            int m = t.Length;
            int[] v0 = new int[m + 1];
            int[] v1 = new int[m + 1];

            for (int i = 0; i <= m; i++) v0[i] = i;

            for (int i = 0; i < n; i++)
            {
                v1[0] = i + 1;
                for (int j = 0; j < m; j++)
                {
                    int cost = (s[i] == t[j]) ? 0 : 1;
                    v1[j + 1] = Math.Min(Math.Min(v1[j] + 1, v0[j + 1] + 1), v0[j] + cost);
                }
                Array.Copy(v1, v0, m + 1);
            }

            return v0[m];
        }
    }
}