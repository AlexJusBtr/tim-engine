using System;
using System.IO;

namespace TIMShield
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine("== TIMShield: Sinusoidal String Embedding Fuzzy Search ==");
            Console.WriteLine("Enter data type (Domain/Malware):");
            string dataType = Console.ReadLine()?.Trim().ToUpperInvariant();
            bool isMalware = dataType == "MALWARE";

            Console.WriteLine("Press a key to start...");
            Console.ReadKey();

            string fileName = isMalware ? "malware_signatures.csv" : "domain_list.csv";
            string dataFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, fileName);

            if (!File.Exists(dataFilePath))
            {
                Console.WriteLine($"\nCRITICAL ERROR: Could not find '{fileName}'.");
                Console.WriteLine("Make sure 'Copy to Output Directory' is set to 'Copy if newer' in the file properties.");
                Console.WriteLine("Press a key to exit...");
                Console.ReadKey();
                return;
            }

            var dataItems = DataLoader.LoadData(dataFilePath, isMalware);
            if (dataItems.Count == 0)
            {
                Console.WriteLine("Error: No valid data found in the file. Please check the file format and content.");
                Console.WriteLine("Press a key to exit...");
                Console.ReadKey();
                return;
            }

            Console.WriteLine($"\nLoaded {dataItems.Count:N0} {(isMalware ? "malware signatures" : "domains")} from file.");

            string query = isMalware ? "a1b2c3d4e5f6" : "p@ypal.com";
            Console.WriteLine($"Query: {query}");

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

            Console.WriteLine("Press a key to exit...");
            Console.ReadKey();
        }
    }
}
