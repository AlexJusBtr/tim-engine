using System;

namespace TIMShield
{
    public class EncodedRecord
    {
        public float[] Embedding { get; set; }
        public int[] TopDimensions { get; set; }
    }

    public class SearchResult
    {
        public string MatchedItem { get; set; }
        public bool IsExactMatch { get; set; }
        public float CosineSimilarity { get; set; }
        public TimeSpan SearchTime { get; set; }
        public int CandidatesEvaluated { get; set; }
    }
}
