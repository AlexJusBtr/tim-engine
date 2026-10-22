using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace TIMShield
{
    public static class DatasetGenerator
    {
        private static readonly string[] Roots = 
        {
            "paypal", "google", "microsoft", "apple", "amazon", "netflix", "facebook", 
            "instagram", "twitter", "linkedin", "github", "gitlab", "dropbox", "salesforce", 
            "slack", "zoom", "spotify", "steam", "discord", "reddit", "chase", "bankofamerica", 
            "wellsfargo", "citi", "standardbank", "fnb", "absa", "nedbank", "capitecbank"
        };

        private static readonly string[] Prefixes = 
        {
            "secure", "login", "account", "auth", "portal", "verify", "update", "web", "app", "cloud"
        };

        private static readonly string[] Tlds = 
        {
            ".com", ".net", ".org", ".io", ".co.za", ".app", ".info", ".online"
        };

        public static void GenerateDataset(string outputPath, int targetCount = 10000)
        {
            var random = new Random(42);
            var uniqueDomains = new HashSet<string>(targetCount);
            const string typoChars = "abcdefghijklmnopqrstuvwxyz0123456789-";

            // 1. Legitimate base and prefixed domains
            foreach (var root in Roots)
            {
                foreach (var tld in Tlds)
                {
                    uniqueDomains.Add($"{root}{tld}");
                    foreach (var prefix in Prefixes)
                    {
                        uniqueDomains.Add($"{prefix}-{root}{tld}");
                        uniqueDomains.Add($"{root}-{prefix}{tld}");
                    }
                }
            }

            // 2. Synthesize typos and variants until target count is reached
            while (uniqueDomains.Count < targetCount)
            {
                int mode = random.Next(4);
                string root = Roots[random.Next(Roots.Length)];
                string tld = Tlds[random.Next(Tlds.Length)];

                switch (mode)
                {
                    case 0: // Single-character typo substitution
                        int idx = random.Next(root.Length);
                        char sub = typoChars[random.Next(typoChars.Length)];
                        var chars = root.ToCharArray();
                        chars[idx] = sub;
                        uniqueDomains.Add($"{new string(chars)}{tld}");
                        break;

                    case 1: // Random sub-token prefix
                        string token = GenerateRandomToken(random, random.Next(4, 8));
                        uniqueDomains.Add($"{token}-{root}{tld}");
                        break;

                    case 2: // Compound root
                        string root2 = Roots[random.Next(Roots.Length)];
                        uniqueDomains.Add($"{root}-{root2}{tld}");
                        break;

                    default: // Arbitrary domain token
                        string randToken = GenerateRandomToken(random, random.Next(6, 12));
                        uniqueDomains.Add($"{randToken}{tld}");
                        break;
                }
            }

            File.WriteAllLines(outputPath, uniqueDomains, Encoding.UTF8);
            Console.WriteLine($"[DatasetGenerator] Wrote {uniqueDomains.Count:N0} records to '{outputPath}'.");
        }

        private static string GenerateRandomToken(Random random, int length)
        {
            const string alphabet = "abcdefghijklmnopqrstuvwxyz";
            var sb = new StringBuilder(length);
            for (int i = 0; i < length; i++)
            {
                sb.Append(alphabet[random.Next(alphabet.Length)]);
            }
            return sb.ToString();
        }
    }
}