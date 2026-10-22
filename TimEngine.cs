using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Cloo;

namespace TIMShield
{
    /// <summary>
    /// Sinusoidal Positional String Embedding & Inverted-Index Fuzzy Search Engine.
    /// Encodes strings into low-dimensional unit vectors using Random Fourier Features (RFF)
    /// over character values and character positions, using OpenCL for batch GPU encoding
    /// and a top-dimension inverted index for coarse candidate filtering.
    /// </summary>
    public class TimEngine
    {
        private readonly int _dimensions;
        private readonly int _bucketsPerRecord;
        private readonly float _minSimilarityThreshold = 0.55f;
        private readonly int _minSharedBuckets;
        private readonly List<string> _items;
        private readonly List<EncodedRecord> _records;
        private readonly Dictionary<int, List<int>> _invertedIndex;
        private readonly float[] _baseFrequencies;
        private readonly float[] _phaseOffsets;
        private readonly Random _random = new Random(42);
        private ComputeContext _context;
        private ComputeCommandQueue _queue;
        private ComputeProgram _program;
        private readonly bool _isMalware;
        private readonly bool _verbose;
        private int _maxLength;

        private void Debug(string msg)
        {
            if (_verbose) Console.WriteLine(msg);
        }

        private void Info(string msg)
        {
            Console.WriteLine(msg);
        }

        public TimEngine(List<string> items, bool isMalware, bool verbose = false)
        {
            _items = items ?? throw new ArgumentNullException(nameof(items));
            _isMalware = isMalware;
            _verbose = verbose;

            _dimensions = Math.Min(32, Math.Max(16, (int)Math.Log2(Math.Max(2, items.Count)) * 2));
            _bucketsPerRecord = Math.Min(8, Math.Max(4, (int)Math.Log10(Math.Max(10, items.Count)) + 2));
            _minSharedBuckets = Math.Max(1, _bucketsPerRecord / 3);

            _records = new List<EncodedRecord>(items.Count);
            _invertedIndex = new Dictionary<int, List<int>>(_dimensions);
            _baseFrequencies = new float[_dimensions];
            _phaseOffsets = new float[_dimensions];

            for (int d = 0; d < _dimensions; d++)
            {
                _baseFrequencies[d] = (float)(_random.NextDouble() * 30.0 + 5.0);
                _phaseOffsets[d] = (float)(_random.NextDouble() * 2.0 * Math.PI);
                _invertedIndex[d] = new List<int>(Math.Max(1000, items.Count / 5));
            }

            InitializeOpenCL();
        }

        private void InitializeOpenCL()
        {
            if (ComputePlatform.Platforms.Count == 0)
            {
                throw new InvalidOperationException("No OpenCL platforms available.");
            }

            var platform = ComputePlatform.Platforms.FirstOrDefault(p => p.Name.Contains("NVIDIA"))
                           ?? ComputePlatform.Platforms[0];

            var devices = platform.Devices.Where(d => d.Type.HasFlag(ComputeDeviceTypes.Gpu)).ToList();
            if (!devices.Any())
            {
                devices = platform.Devices.Where(d => d.Type.HasFlag(ComputeDeviceTypes.Cpu)).ToList();
                Debug("Warning: No GPU found, falling back to CPU: " + devices.FirstOrDefault()?.Name);
            }

            if (!devices.Any())
            {
                throw new InvalidOperationException("No OpenCL devices available.");
            }

            var properties = new ComputeContextPropertyList(platform);
            _context = new ComputeContext(devices, properties, null, IntPtr.Zero);
            _queue = new ComputeCommandQueue(_context, devices[0], ComputeCommandQueueFlags.None);
            Info($"Using OpenCL device: {devices[0].Name} (Type: {devices[0].Type}, MaxWorkGroupSize: {devices[0].MaxWorkGroupSize})");

            string kernelSource = GetOpenClKernelSource();
            _program = new ComputeProgram(_context, kernelSource);
            try
            {
                _program.Build(null, null, null, IntPtr.Zero);
                Info("OpenCL batch encoding kernel compiled successfully.");
            }
            catch (ComputeException)
            {
                string buildLog = string.Join("\n", _context.Devices.Select(d => $"Device {d.Name}: {_program.GetBuildLog(d)}"));
                Debug($"OpenCL build error details:\n{buildLog}");
                throw;
            }
        }

        public void BuildIndex()
        {
            if (!_items.Any())
            {
                throw new InvalidOperationException("No items provided.");
            }

            var stopwatch = Stopwatch.StartNew();
            _maxLength = _items.Max(s => s.Length);
            var paddedItems = _items.Select(s => s.PadRight(_maxLength, '\0')).ToArray();

            var encoded = _items.Count < 100
                ? EncodeBatchCpu(paddedItems, _maxLength)
                : EncodeBatchGpu(paddedItems, _maxLength);

            for (int idx = 0; idx < encoded.Length; idx++)
            {
                _records.Add(encoded[idx]);
                foreach (int dim in encoded[idx].TopDimensions)
                {
                    if (dim >= 0 && dim < _dimensions)
                    {
                        _invertedIndex[dim].Add(idx);
                    }
                }
            }

            foreach (var list in _invertedIndex.Values)
            {
                list.TrimExcess();
            }
            _records.TrimExcess();

            stopwatch.Stop();
            Info($"Index built: {_records.Count:N0} records encoded and indexed in {stopwatch.ElapsedMilliseconds:F0} ms");
        }

        private EncodedRecord[] EncodeBatchGpu(string[] paddedItems, int maxLength)
        {
            int numItems = paddedItems.Length;
            Info($"Batch-encoding {numItems:N0} strings on GPU...");

            var flatInputBytes = Encoding.ASCII.GetBytes(string.Concat(paddedItems));
            long embeddingsSize = (long)numItems * _dimensions * sizeof(float);
            long maxMemAlloc = _context.Devices[0].MaxMemoryAllocationSize;

            if (embeddingsSize > maxMemAlloc)
            {
                throw new InvalidOperationException("Buffer size exceeds GPU memory limit.");
            }

            ComputeBuffer<byte> inputBuffer = null;
            ComputeBuffer<float> embeddingsBuffer = null;
            ComputeBuffer<float> frequenciesBuffer = null;
            ComputeBuffer<float> offsetsBuffer = null;
            ComputeKernel kernel = null;

            try
            {
                inputBuffer = new ComputeBuffer<byte>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, flatInputBytes);
                embeddingsBuffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.WriteOnly, numItems * _dimensions);
                frequenciesBuffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, _baseFrequencies);
                offsetsBuffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, _phaseOffsets);

                kernel = _program.CreateKernel("encodeKernel");
                kernel.SetMemoryArgument(0, inputBuffer);
                kernel.SetMemoryArgument(1, embeddingsBuffer);
                kernel.SetMemoryArgument(2, frequenciesBuffer);
                kernel.SetMemoryArgument(3, offsetsBuffer);
                kernel.SetValueArgument(4, numItems);
                kernel.SetValueArgument(5, maxLength);
                kernel.SetValueArgument(6, _dimensions);

                long maxWorkItems = Math.Min(_context.Devices[0].MaxWorkGroupSize, numItems);
                for (long offset = 0; offset < numItems; offset += maxWorkItems)
                {
                    long batchSize = Math.Min(maxWorkItems, numItems - offset);
                    _queue.Execute(kernel, new long[] { offset }, new long[] { batchSize }, null, null);
                }
                _queue.Finish();

                float[] flatEmbeddings = new float[numItems * _dimensions];
                _queue.ReadFromBuffer(embeddingsBuffer, ref flatEmbeddings, true, null);

                var results = new EncodedRecord[numItems];
                for (int idx = 0; idx < numItems; idx++)
                {
                    var vec = new float[_dimensions];
                    Array.Copy(flatEmbeddings, idx * _dimensions, vec, 0, _dimensions);
                    results[idx] = new EncodedRecord
                    {
                        Embedding = vec,
                        TopDimensions = SelectTopDimensions(vec, _bucketsPerRecord)
                    };
                }
                return results;
            }
            finally
            {
                inputBuffer?.Dispose();
                embeddingsBuffer?.Dispose();
                frequenciesBuffer?.Dispose();
                offsetsBuffer?.Dispose();
                kernel?.Dispose();
            }
        }

        private EncodedRecord[] EncodeBatchCpu(string[] paddedItems, int maxLength)
        {
            int numItems = paddedItems.Length;
            var results = new EncodedRecord[numItems];
            for (int idx = 0; idx < numItems; idx++)
            {
                results[idx] = EncodeSingleCpu(paddedItems[idx], maxLength);
            }
            return results;
        }

        private EncodedRecord EncodeSingleCpu(string input, int maxLength)
        {
            var embedding = new float[_dimensions];
            int validChars = 0;

            for (int pos = 0; pos < maxLength && pos < input.Length && input[pos] != '\0'; pos++)
            {
                char c = input[pos];
                float charValue = _isMalware ? (float)c : DataLoader.DomainAlphabet.IndexOf(c);
                if (charValue < 0.0f) continue;

                validChars++;
                float charFrequency = charValue * (_isMalware ? 0.1f : 0.2f);
                float positionPhase = pos * (float)Math.PI / maxLength;
                float lengthScale = maxLength < 50 ? 1.5f : 1.0f;
                float posWeight = (1.0f + pos * 0.05f) * lengthScale;

                for (int d = 0; d < _dimensions; d++)
                {
                    float feature = (float)Math.Sin(_baseFrequencies[d] * charFrequency + _phaseOffsets[d] + positionPhase);
                    embedding[d] += feature * posWeight;
                }
            }

            if (validChars == 0)
            {
                return new EncodedRecord { Embedding = embedding, TopDimensions = Array.Empty<int>() };
            }

            float magnitude = 0.0f;
            for (int d = 0; d < _dimensions; d++)
            {
                magnitude += embedding[d] * embedding[d];
            }
            magnitude = (float)Math.Sqrt(magnitude);

            if (magnitude < 0.0001f)
            {
                return new EncodedRecord { Embedding = new float[_dimensions], TopDimensions = Array.Empty<int>() };
            }

            for (int d = 0; d < _dimensions; d++)
            {
                embedding[d] /= magnitude;
            }

            return new EncodedRecord
            {
                Embedding = embedding,
                TopDimensions = SelectTopDimensions(embedding, _bucketsPerRecord)
            };
        }

        /// <summary>
        /// Selects the top-K dimensions by absolute magnitude in the normalized embedding vector
        /// to act as coarse inverted-index bucket keys.
        /// </summary>
        private static int[] SelectTopDimensions(float[] embedding, int k)
        {
            return Enumerable.Range(0, embedding.Length)
                .OrderByDescending(d => Math.Abs(embedding[d]))
                .Take(k)
                .ToArray();
        }

        private string GetOpenClKernelSource()
        {
            return _isMalware ? @"
                __kernel void encodeKernel(
                    __global const char* input,
                    __global float* embeddings,
                    __global const float* baseFrequencies,
                    __global const float* phaseOffsets,
                    int itemCount,
                    int maxLength,
                    int Dimensions)
                {
                    int idx = get_global_id(0);
                    if (idx >= itemCount) return;

                    float vec[32];
                    for (int d = 0; d < Dimensions; d++) vec[d] = 0.0f;

                    int validChars = 0;
                    for (int pos = 0; pos < maxLength; pos++)
                    {
                        char c = input[idx * maxLength + pos];
                        if (c == ' ') break;

                        float charValue = (float)c;
                        validChars++;
                        float charFrequency = charValue * 0.1f;
                        float positionPhase = pos * 3.1415926f / maxLength;
                        float lengthScale = maxLength < 50 ? 1.5f : 1.0f;
                        float posWeight = (1.0f + pos * 0.05f) * lengthScale;

                        for (int d = 0; d < Dimensions; d++)
                        {
                            float feature = sin(baseFrequencies[d] * charFrequency + phaseOffsets[d] + positionPhase);
                            vec[d] += feature * posWeight;
                        }
                    }

                    float magnitude = 0.0f;
                    for (int d = 0; d < Dimensions; d++)
                        magnitude += vec[d] * vec[d];
                    magnitude = sqrt(magnitude);

                    if (validChars == 0 || magnitude < 0.0001f)
                    {
                        for (int d = 0; d < Dimensions; d++) embeddings[idx * Dimensions + d] = 0.0f;
                        return;
                    }

                    for (int d = 0; d < Dimensions; d++)
                        embeddings[idx * Dimensions + d] = vec[d] / magnitude;
                }"
            : @"
                __kernel void encodeKernel(
                    __global const char* input,
                    __global float* embeddings,
                    __global const float* baseFrequencies,
                    __global const float* phaseOffsets,
                    int itemCount,
                    int maxLength,
                    int Dimensions)
                {
                    int idx = get_global_id(0);
                    if (idx >= itemCount) return;

                    float vec[32];
                    for (int d = 0; d < Dimensions; d++) vec[d] = 0.0f;

                    int validChars = 0;
                    __constant char alphabet[] = ""abcdefghijklmnopqrstuvwxyz0123456789-.@"";
                    const int alphabetSize = 39;
                    int inputOffset = idx * maxLength;

                    for (int pos = 0; pos < maxLength; pos++)
                    {
                        char c = input[inputOffset + pos];
                        if (c == ' ') break;

                        float charValue = -1.0f;
                        for (int i = 0; i < alphabetSize; i++)
                        {
                            if (c == alphabet[i])
                            {
                                charValue = (float)i;
                                break;
                            }
                        }
                        if (charValue < 0.0f) continue;

                        validChars++;
                        float charFrequency = charValue * 0.2f;
                        float positionPhase = pos * 3.1415926f / maxLength;
                        float lengthScale = maxLength < 50 ? 1.5f : 1.0f;
                        float posWeight = (1.0f + pos * 0.05f) * lengthScale;

                        for (int d = 0; d < Dimensions; d++)
                        {
                            float feature = sin(baseFrequencies[d] * charFrequency + phaseOffsets[d] + positionPhase);
                            vec[d] += feature * posWeight;
                        }
                    }

                    float magnitude = 0.0f;
                    for (int d = 0; d < Dimensions; d++)
                        magnitude += vec[d] * vec[d];
                    magnitude = sqrt(magnitude);

                    if (validChars == 0 || magnitude < 0.0001f)
                    {
                        for (int d = 0; d < Dimensions; d++) embeddings[idx * Dimensions + d] = 0.0f;
                        return;
                    }

                    for (int d = 0; d < Dimensions; d++)
                        embeddings[idx * Dimensions + d] = vec[d] / magnitude;
                }";
        }

        public List<SearchResult> Search(string query, int topK)
        {
            var stopwatch = Stopwatch.StartNew();

            string cleanedQuery = _isMalware
                ? query.Trim().ToLowerInvariant()
                : new string(query.ToLowerInvariant().Where(c => DataLoader.DomainAlphabet.Contains(c)).ToArray());

            // Encode single query string directly on CPU (avoids PCIe buffer allocation overhead for 1 item)
            var queryRecord = EncodeSingleCpu(cleanedQuery, _maxLength);

            // Coarse candidate filtering via inverted index
            var sharedBucketCounts = new int[_items.Count];
            var candidateSet = new List<int>(_items.Count / 10);

            foreach (int dim in queryRecord.TopDimensions)
            {
                if (_invertedIndex.TryGetValue(dim, out var bucket))
                {
                    foreach (int idx in bucket)
                    {
                        sharedBucketCounts[idx]++;
                        if (sharedBucketCounts[idx] == _minSharedBuckets)
                        {
                            candidateSet.Add(idx);
                        }
                    }
                }
            }

            double evaluatedPercentage = (double)candidateSet.Count / _items.Count * 100.0;
            Info($"Candidates passing bucket filter: {candidateSet.Count:N0} / {_items.Count:N0} ({evaluatedPercentage:F2}% of dataset)");

            var topCandidates = new List<(float Similarity, int Index)>();
            int candidatesAboveThreshold = 0;

            foreach (int idx in candidateSet)
            {
                float cosineSim = DotProduct(queryRecord.Embedding, _records[idx].Embedding);
                if (cosineSim >= _minSimilarityThreshold)
                {
                    candidatesAboveThreshold++;
                    topCandidates.Add((cosineSim, idx));
                }
            }

            topCandidates.Sort((a, b) => b.Similarity.CompareTo(a.Similarity));
            if (topCandidates.Count > topK)
            {
                topCandidates.RemoveRange(topK, topCandidates.Count - topK);
            }

            stopwatch.Stop();

            var results = new List<SearchResult>(topCandidates.Count);
            foreach (var candidate in topCandidates)
            {
                results.Add(new SearchResult
                {
                    MatchedItem = _items[candidate.Index],
                    IsExactMatch = _items[candidate.Index] == cleanedQuery,
                    CosineSimilarity = candidate.Similarity,
                    SearchTime = stopwatch.Elapsed,
                    CandidatesEvaluated = candidateSet.Count
                });
            }

            return results;
        }

        /// <summary>
        /// Computes cosine similarity between two unit-normalized vectors via dot product.
        /// </summary>
        private float DotProduct(float[] a, float[] b)
        {
            float sum = 0.0f;
            for (int d = 0; d < _dimensions; d++)
            {
                sum += a[d] * b[d];
            }
            return Math.Max(-1.0f, Math.Min(1.0f, sum));
        }
    }
}
