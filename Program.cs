using System;
using System.IO;

namespace TIMShield
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine("== TIM: Sinusoidal String Embedding Fuzzy Search ==");
            Console.WriteLine("Select data category:");
            Console.WriteLine(" 1. Domain Names");
            Console.WriteLine(" 2. Malware Signatures (Hex/Hashes)");
            Console.Write("Choice [1/2]: ");
            string choice = Console.ReadLine()?.Trim() ?? "1";
            bool isMalware = choice == "2";

            string defaultFileName = isMalware ? "malware_signatures.csv" : "domain_list.csv";

            // Allow choosing between default, generated, or custom file
            Console.WriteLine($"\nEnter CSV filename or press Enter for default ('{defaultFileName}'):");
            Console.WriteLine(" Tip: Type 'gen' to auto-generate a fresh 10,000-domain dataset.");
            Console.Write("Filename: ");
            string inputFileName = Console.ReadLine()?.Trim() ?? "";

            string fileName;
            if (string.Equals(inputFileName, "gen", StringComparison.OrdinalIgnoreCase))
            {
                fileName = "domain_list_10k.csv";
                string fullPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, fileName);
                DatasetGenerator.GenerateDataset(fullPath, targetCount: 10000);
            }
            else if (!string.IsNullOrEmpty(inputFileName))
            {
                fileName = inputFileName;
            }
            else
            {
                fileName = defaultFileName;
            }

            string dataFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, fileName);

            // Check if file exists in BaseDirectory or project root
            if (!File.Exists(dataFilePath))
            {
                // Fallback check in current directory
                if (File.Exists(fileName))
                {
                    dataFilePath = Path.GetFullPath(fileName);
                }
                else
                {
                    Console.WriteLine($"\nCRITICAL ERROR: Could not find '{fileName}'.");
                    Console.WriteLine($"Expected location: {dataFilePath}");
                    Console.WriteLine("Press a key to exit...");
                    Console.ReadKey();
                    return;
                }
            }

            var dataItems = DataLoader.LoadData(dataFilePath, isMalware);
            if (dataItems.Count == 0)
            {
                Console.WriteLine("Error: No valid data found in the file.");
                Console.WriteLine("Press a key to exit...");
                Console.ReadKey();
                return;
            }

            Console.WriteLine($"\nLoaded {dataItems.Count:N0} records from '{fileName}'.");

            // Query selection
            string defaultQuery = isMalware ? "a1b2c3d4e5f6" : "p@ypal.com";
            Console.WriteLine($"\nEnter query string or press Enter for default ('{defaultQuery}'):");
            Console.Write("Query: ");
            string enteredQuery = Console.ReadLine()?.Trim() ?? "";
            string query = string.IsNullOrEmpty(enteredQuery) ? defaultQuery : enteredQuery;

            // Build Index
            var engine = new TimEngine(dataItems, isMalware, verbose: false);
            try
            {
                engine.BuildIndex();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error building index: {ex.Message}");
                Console.WriteLine("Press a key to exit...");
                Console.ReadKey();
                return;
            }

            // Execute Search
            Console.WriteLine("\nRunning fuzzy search...");
            var results = engine.Search(query, topK: 5);

            Console.WriteLine("\n== Search Results ==");
            foreach (var result in results)
            {
                Console.WriteLine($"Match: {result.MatchedItem}");
                Console.WriteLine($"Cosine Similarity: {result.CosineSimilarity:F4}");
                Console.WriteLine($"Exact Match: {result.IsExactMatch}");
                Console.WriteLine($"Search Time: {result.SearchTime.TotalMilliseconds:F2} ms");
                Console.WriteLine($"Candidates Evaluated: {result.CandidatesEvaluated:N0}");
                Console.WriteLine("---");
            }

            // Run Recall Benchmark
            Console.WriteLine("\nRun recall benchmark against Levenshtein? (y/n)");
            if (Console.ReadLine()?.Trim().ToLowerInvariant() == "y")
            {
                var benchmark = new RecallBenchmark(dataItems, engine);
                benchmark.Run(numTestQueries: 100, topK: 5);
            }

            // At the end of Program.cs, before Console.ReadKey():
            Console.WriteLine("\nRun concurrent throughput (QPS) benchmark? (y/n)");
                if (Console.ReadLine()?.Trim().ToLowerInvariant() == "y")
                {
                    var throughput = new ThroughputBenchmark(engine, dataItems);
                    throughput.Run(totalQueries: 5000, topK: 5);
                }
            Console.WriteLine("Press a key to exit...");
            Console.ReadKey();
        }
    }
}