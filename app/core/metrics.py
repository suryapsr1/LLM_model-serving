from prometheus_client import Counter, Histogram, Gauge

# 1. Gauge for active concurrent streaming connections
ACTIVE_STREAMS = Gauge(
    "llm_active_streams", 
    "Number of currently active streaming sessions"
)

# 2. Stream chunks are not necessarily individual model tokens.
OUTPUT_CHUNKS = Counter(
    "llm_output_chunks_total",
    "Number of non-empty text chunks emitted to streaming clients",
    ["engine"]
)

# 3. Time from stream start to the first non-empty text chunk.
TIME_TO_FIRST_CHUNK = Histogram(
    "llm_time_to_first_chunk_seconds",
    "Latency from stream start to the first non-empty text chunk",
    ["engine"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# 4. Histogram tracking total generation duration
TOTAL_GENERATION_DURATION = Histogram(
    "llm_generation_duration_seconds",
    "Total duration of a streaming generation request",
    ["engine"]
)
