# Benchmark results

## M5 Pro / Qwen3-8B-4bit / 2026-09-06

The first controlled MLX matrix used AC power, macOS High Power mode, one warmup,
and five measured trials per configuration. The exact model revision and all raw
trials are stored in the
[`machine-readable record`](2026-09-06-m5-pro-qwen3-8b-4bit.json). See the
[`full methodology and results report`](2026-09-06-m5-pro-qwen3-8b-4bit.md) for
reproduction instructions, interpretation, and limitations.

| Prompt | Generate | Median prompt tok/s | Median generation tok/s | Peak memory |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 950.720 | 59.971 | 4.923 GB |
| 128 | 256 | 960.152 | 59.393 | 4.923 GB |
| 256 | 128 | 1254.467 | 59.593 | 5.064 GB |
| 256 | 256 | 972.618 | 59.089 | 5.064 GB |
| 512 | 128 | 1226.206 | 57.363 | 5.217 GB |

Generation throughput was stable, with coefficients of variation from 0.50% to
1.29%. Prompt throughput varied substantially and should be repeated after the
battery is charged and the system has cooled. The battery was actively charging
from 29% to 33% during this run.
