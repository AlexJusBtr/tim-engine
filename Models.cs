using System;

namespace TIMShield
{
    public class EncodedRecord
    {
        public float[] Embedding { get; set; } = Array.Empty<float>();
        public int[] TopDimensions { get; set; } = Array.Empty<int>();
    }

    public class SearchResult
    {
        public string MatchedItem { get; set; } = string.Empty;
        public bool IsExactMatch { get; set; }
        public float CosineSimilarity { get; set; }
        public TimeSpan SearchTime { get; set; }
        public int CandidatesEvaluated { get; set; }
    }
}