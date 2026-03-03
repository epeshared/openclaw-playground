# session-memory-stress

This is a small stress harness for **session transcript indexing** (and therefore embeddings) in a *real* OpenClaw Gateway process.

It spawns an isolated gateway with an isolated config/state dir, creates a configurable number of sessions ("rooms"), then runs multiple WebSocket clients that call `sessions.reset` + `chat.send` (default) or `chat.inject` at a controlled rate.

---

下面是更“落地”的说明：它在模拟什么场景、跑起来之后请求是怎么一路走到 embedding 的，以及如何用 `memory_search` 验证确实发生了 session transcript indexing。

## 模拟的场景（它在压什么）

这个脚本模拟的是「多个前端/终端同时连到 Gateway，在多个会话里持续产生对话内容」的真实负载。

当你开启 session transcript indexing（脚本会在 spawned gateway 的配置里打开）时，Gateway 会把会话 transcript（jsonl）增量索引到内存索引（SQLite）里；索引过程会进行 chunking + embedding（本地或远端），从而形成典型的 embedding/向量检索压力。

## 调用链路：从 RPC 到 embedding（发生了什么）

以默认“脚本自己 spawn 一个 gateway”的路径为例（`--spawnGateway=1`）：

1. 脚本写入一个临时配置到 `OPENCLAW_CONFIG_PATH`，并设置 `OPENCLAW_STATE_DIR` 指向一个隔离的 state 目录。
2. 脚本启动 `node ./openclaw.mjs gateway run ...`。
3. 脚本连接 WebSocket（GatewayClient）并对每个 room 执行 `sessions.reset`，确保 session 存在。
4. 压测阶段通过两种方式之一写入会话内容：
  - `chat.inject`：直接“追加 transcript 消息”（更接近“只写 transcript、尽量少 agent 运行开销”的路径）。
  - `chat.send`：按正常收消息流程进入（可能触发 agent run）。脚本默认会 `chat.abort` 立即终止，避免 agent 侧计算堆积，但这也可能让 transcript 侧可索引内容变少/变慢。
5. 当 transcript 文件发生变化时，Gateway 会触发一个“transcript 更新事件”（进程内通知）。
6. `memory_search` 的 session 索引管理器监听该事件，做一个 ~5s 的 debounce，然后根据增量阈值（`deltaBytes`/`deltaMessages`）决定是否触发 `sync()`。
7. `sync()` 会读取对应的 session transcript（jsonl），将文本 chunk 成多段（`chunkTokens`/`chunkOverlap`），然后对每个 chunk 调用 embedding provider：
  - `--embeddingProvider=local`：本地 ggml/llama 路径（CPU 热点通常会落在 `libggml-*` 的矩阵乘等内核上）
  - `--embeddingProvider=openai|gemini|voyage|mistral`：远端 embedding API
8. embedding 结果写入内存索引 SQLite（默认在 state dir 下的 `memory/<agentId>.sqlite`），并在 `memory_search(query)` 时用于向量检索/混合检索（hybrid）。

一句话：这个脚本不是“模拟调用 embedding API”，而是在驱动真实 Gateway 的 transcript→索引→embedding 的完整链路。

## 如何确保”确实调用到了 embedding”

建议使用 `chat.inject` + 本地 embedding 来确保链路足够”硬”，且热点更容易聚焦在 embedding 计算。

注意：如果 embedding provider 配置不完整（例如远端缺 key，或本地模型/依赖缺失），OpenClaw 可能退化到 **FTS-only**（关键词检索），这会让”embedding 压力”消失或变小。

## 关键参数

| 参数 | 说明 | 默认值 |
|-----------|-------------|--------|
| `--clients=N` | WebSocket 客户端数量 | 1 |
| `--rooms=M` | 会话（房间）数量 | 1 |
| `--messageRate` | 每个客户端每秒消息数 | 5 |
| `--durationMs` | 测试持续时间（毫秒） | 30000 |
| `--messageChars` | 每条消息的字符数 | 8192 |
| `--chatMethod=inject\|send` | 写入方式（`inject` 为纯 transcript 追加，`send` 为完整流程） | send |
| `--embeddingProvider=local\|openai\|gemini\|voyage\|mistral` | Embedding 后端 | openai |
| `--chunkTokens` | 每个 chunk 的 token 数（越小产生越多 embedding 调用） | 400 |
| `--chunkOverlap` | chunk 之间的重叠 | 80 |
| `--deltaBytes` | 触发索引的最小字节变化 | 1 |
| `--deltaMessages` | 触发索引的最小消息数变化 | 1 |
| `--cache=0\|1` | 启用 embedding 缓存 | 1 |
| `--vector=0\|1` | 启用 sqlite-vec 进行向量搜索 | 0 |
| `--verifySearch=0\|1` | 压测后执行验证查询 | 0 |
| `--verifyWaitMs` | 验证前等待时间（默认 6500ms，考虑约 5s debounce） | 6500 |
| `--gatewayLogs=0\|1` | 将 gateway 日志输出到 stderr | 1 |

## Run

From the repo root:

```bash
npx --yes tsx@4.21.0 openclaw-playground/embedding/session-memory-stress/run-with-gateway.ts \
  --clients=4 --rooms=2 --durationMs=30000 --messageRate=10 --messageChars=16384 \
  --embeddingProvider=openai --verifySearch=1
```

If gateway logs are too noisy, add `--gatewayLogs=0`.

### 使用“外部已启动网关”（脚本不负责启动/关闭）

如果你已经手动启动了 OpenClaw Gateway（例如你想用 `numactl`/`taskset` 固定绑核，或想复用一个长期运行的网关进程），可以让脚本只做压测调用，不 spawn 子进程网关：

```bash
npx --yes tsx@4.21.0 openclaw-playground/embedding/session-memory-stress/run-with-gateway.ts \
  --spawnGateway=0 \
  --url=ws://127.0.0.1:18789 \
  --token=<你的token> \
  --clients=4 --rooms=2 --durationMs=30000 --messageRate=10 --messageChars=16384
```

说明：

- 该模式下脚本不会写入 `OPENCLAW_STATE_DIR` / `OPENCLAW_CONFIG_PATH` 来控制网关，也不会负责关闭网关进程。
- 你需要保证 `--url`/`--token` 与你手动启动的网关一致，并且该网关本身的配置已开启你想测的 memorySearch/session 索引与 embedding provider。

### 一个最小可验证（Recommended）命令

这个命令的目标是：短时间内写入 transcript，并且 `memory_search(<marker>)` 能命中（证明索引完成）。

```bash
npx --yes tsx@4.21.0 openclaw-playground/embedding/session-memory-stress/run-with-gateway.ts \
  --clients=1 --rooms=1 --durationMs=2000 --messageRate=1 --messageChars=2048 \
  --chatMethod=inject \
  --embeddingProvider=local --cache=0 --vector=1 --deltaBytes=1 --deltaMessages=1 \
  --verifySearch=1 --verifyWaitMs=6500 --verifyPollEveryMs=1000 --verifyTimeoutMs=60000 \
  --gatewayLogs=0
```

脚本会打印：

- `marker=<token>`：本轮压测的唯一标记
- verify 阶段的 `results=<n>`：命中数量（>0 说明索引里已经有 session chunk）

### Using `chat.send` vs `chat.inject`

- Default is `chat.send` (`--chatMethod=send`).
- `chat.send` requires an `idempotencyKey` per message and may start an agent run. The harness can immediately abort each run with `--sendAbort=1` (default) to reduce background work.
- If you want a pure “append transcript” path with less agent overhead, use `--chatMethod=inject`.

补充说明（为什么你可能会看到 `chat.send` 更难“稳定打到索引”）：

- `chat.send` 的语义更像“正常收消息并可能驱动 agent run”。如果你立刻 `chat.abort`，可能会减少/延后 assistant 侧 transcript 写入，从而让 session 索引触发变得不稳定。
- `chat.inject` 是更直接的“追加 transcript + 触发 transcript update 事件”的路径，用来做“纯索引/embedding 压测”更可控。

### Make embeddings the bottleneck

Switch `--embeddingProvider`:

- `--embeddingProvider=local` (may require model download / native deps)
- `--embeddingProvider=openai` / `gemini` / `voyage` / `mistral` (requires credentials)

Useful knobs:

- `--embeddingModel=...`
- `--remoteBaseUrl=... --remoteApiKey=...` (for OpenAI-compatible endpoints / proxies)
- `--remoteBatch=1 --remoteBatchConcurrency=...` (if your endpoint supports batch)
- `--localModelPath=... --localModelCacheDir=...` (for `--embeddingProvider=local`)

For `chat.send` specifically:

- `--sendAbort=1` aborts runs immediately (recommended for stress)
- `--sendTimeoutMs=2000` sets a small timeout override; `--sendTimeoutMs=0` disables passing the override

By default the harness sets `store.vector.enabled=0` (no sqlite-vec required), so you can focus on the embedding/indexing path.

如果你的目标是“让 perf 热点尽量落在 embedding 计算”，更推荐显式打开向量与小 chunk：

```bash
npx --yes tsx@4.21.0 openclaw-playground/embedding/session-memory-stress/run-with-gateway.ts \
  --clients=1 --rooms=1 --durationMs=30000 --messageRate=10 --messageChars=65536 \
  --chatMethod=inject \
  --embeddingProvider=local --cache=0 --vector=1 \
  --chunkTokens=64 --chunkOverlap=16 --deltaBytes=1 --deltaMessages=1 \
  --verifySearch=1 --verifyWaitMs=6500 --verifyPollEveryMs=2000 --verifyTimeoutMs=180000 \
  --statsEveryMs=2000 --gatewayLogs=0
```

这类配置下，CPU 热点通常会出现大量 `libggml`（比如 GEMM/quantize/rms_norm 等内核）。

If your chosen remote provider is missing API keys, OpenClaw can degrade to **FTS-only** mode (text search) for `memory_search`.

### Stats / verification

- `--statsEveryMs=2000` prints periodic QPS + inject latency percentiles.
- `--probeSearchEveryMs=...` periodically calls `memory_search(<marker>)` during the run to show indexing lag (time to first hit).
- For slow local embeddings, prefer polling: `--verifyPollEveryMs=2000 --verifyTimeoutMs=300000`.

## Notes

- The gateway’s session indexing is debounced (about ~5s), so you won’t see `memory_search` results instantly.
- The script prints `marker=<token>` where `<token>` is letters+digits only; use the token itself as the search term in `memory_search` when debugging.

## 常见坑 / 为什么 verify 可能一直是 0

- **Debounce**：session transcript 更新会有大约 ~5s 的 debounce；`--verifyWaitMs` 太小会导致你在索引还没开始时就查了。
- **Safe reindex 临时库**：索引可能先写入 `memory/<agentId>.sqlite.tmp-<uuid>`，完成后再原子替换为正式库。如果你在替换发生前反复 `memory_search`，可能会持续得到空结果。
- **仅 FTS / provider 不可用**：embedding provider 配置不对时会退化，导致你以为在测 embedding，实际只在测关键词检索。
- **chat.send + 立即 abort**：更像“压 gateway/agent 调度”的压力，不一定稳定触发 transcript 可索引内容；要测索引建议 `chat.inject`。

## Perf 采样（看 embedding 热点）

示例（保存到一个固定文件，方便反复 `perf report`）：

```bash
cd /nvme5/xtang/openclaw-workspace

perf record -F 99 --call-graph dwarf -o /tmp/perf-session-index.data -- \
  npx --yes tsx@4.21.0 openclaw-playground/embedding/session-memory-stress/run-with-gateway.ts \
    --stateDir=/tmp/oc-stress-perf \
    --spawnGateway=1 --gatewayLogs=0 \
    --embeddingProvider=local --cache=0 --vector=1 --chunkTokens=64 --chunkOverlap=16 \
    --deltaBytes=1 --deltaMessages=1 \
    --chatMethod=inject --clients=1 --rooms=1 --durationMs=30000 --messageRate=10 --messageChars=65536 \
    --verifySearch=1 --verifyWaitMs=6500 --verifyPollEveryMs=2000 --verifyTimeoutMs=180000

perf report -i /tmp/perf-session-index.data --stdio --no-children \
  --percent-limit 0.5 --sort comm,dso,symbol --fields overhead,comm,dso,sym --demangle
```

如果链路确实打到了本地 embedding，`perf report` 的前几名通常会落在 `libggml-*` 的矩阵乘/向量点积/量化/归一化等符号上。
