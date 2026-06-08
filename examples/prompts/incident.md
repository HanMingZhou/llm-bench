Summarize this incident report and identify likely inference bottlenecks.

The service runs a tensor-parallel model with high request concurrency. During peak load,
P99 latency rises sharply, GPU memory usage approaches the configured limit, and users report
slow first-token responses. Recent changes include a larger max model length and a higher
GPU memory utilization setting.
