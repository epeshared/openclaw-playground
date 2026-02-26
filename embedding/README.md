# OpenClaw Vector Embedding：位置与功能速查

这份文档总结 OpenClaw 项目中 **vector embedding（向量嵌入）** 相关代码的主要位置、作用，以及从「生成 embedding → 存储向量 → 相似检索 → 上层功能调用」的链路。

> 适用范围：本文聚焦于 *memory / semantic search / recall*（记忆语义检索）这条主线。

---

## TL;DR（你要找的入口）

- **Embedding provider 抽象 + 本地/远程实现**：
  - openclaw-epeshared/src/memory/embeddings.ts
- **内置（builtin）SQLite 记忆索引 + 向量检索**：
  - openclaw-epeshared/src/memory/manager.ts
  - openclaw-epeshared/src/memory/manager-embedding-ops.ts
  - openclaw-epeshared/src/memory/manager-search.ts
  - openclaw-epeshared/src/memory/manager-sync-ops.ts
  - openclaw-epeshared/src/memory/memory-schema.ts
  - openclaw-epeshared/src/memory/sqlite-vec.ts
- **暴露到 Agent 的工具（功能层）**：
  - openclaw-epeshared/src/agents/tools/memory-tool.ts（memory_search / memory_get）
  - openclaw-epeshared/extensions/memory-core/index.ts（默认插件注册工具与 CLI）
- **可选后端：QMD memory（外部索引/embedding 工具链）**：
  - openclaw-epeshared/src/memory/search-manager.ts
  - openclaw-epeshared/src/memory/backend-config.ts
  - openclaw-epeshared/src/memory/qmd-manager.ts
- **可选插件：LanceDB 记忆（独立向量库 + OpenAI embeddings）**：
  - openclaw-epeshared/extensions/memory-lancedb/index.ts
  - openclaw-epeshared/extensions/memory-lancedb/config.ts

---

## 1) 内置（builtin）SQLite 向量 embedding：做什么？

内置 memory 系统的核心目标是：

1. 将 workspace 的 MEMORY.md / memory/*.md（可选：session transcripts）切分为 chunks
2. 为每个 chunk 生成 embedding（本地或远程 provider）
3. 将 chunk 文本、行号范围、embedding 等存入 SQLite
4. 搜索时对 query 生成 embedding，并进行向量相似检索（可选：与 BM25 混合）
5. 把 top 结果以“路径 + 行号 + snippet”的形式返回给 agent（memory_search），并允许安全按行读取（memory_get）

### 1.1 数据结构（SQLite 表）

Schema 由 openclaw-epeshared/src/memory/memory-schema.ts 创建，关键表：

- `chunks`
  - 每个 chunk 一行
  - 重要字段：`path`, `start_line`, `end_line`, `text`, `embedding`, `model`, `updated_at`
  - 注意：`embedding` 这里是 **JSON 文本**（不是 BLOB）

- `embedding_cache`
  - 缓存 chunk 的 embedding，避免 reindex / sync 时重复请求 embeddings
  - 主键为 `(provider, model, provider_key, hash)`

- `chunks_fts`（可选）
  - FTS5 虚拟表，用于 BM25 关键词检索

### 1.2 向量检索加速：sqlite-vec（vec0 虚拟表）

如果启用 `sqlite-vec` 扩展，则会创建 `chunks_vec`：

- `chunks_vec` 是 `vec0(...)` 虚拟表
- 字段：`id TEXT PRIMARY KEY` + `embedding FLOAT[dimensions]`

加载逻辑：

- openclaw-epeshared/src/memory/sqlite-vec.ts 负责 `import("sqlite-vec")` 并 `load(...)`
- openclaw-epeshared/src/memory/manager-sync-ops.ts 负责 lazy-load + timeout，并在维度变化时重建 vec 表

检索逻辑：

- 优先走 SQL：`vec_distance_cosine(v.embedding, ?) AS dist`
- 若 sqlite-vec 不可用，则回退到 JS 端：从 `chunks.embedding` 解析成数组，然后算 `cosineSimilarity` 做 brute-force 排序

### 1.3 Embedding provider（本地/远程）

openclaw-epeshared/src/memory/embeddings.ts 定义：

- `EmbeddingProvider`：`embedQuery(text) -> number[]`，`embedBatch(texts) -> number[][]`
- provider id：`openai | gemini | voyage | mistral | local | auto`

关键点：

- 本地 provider 使用 node-llama-cpp 的 embedding context，并对向量做 **L2 normalize**
- `auto` 会优先尝试 local（当存在可用的本地模型路径），再尝试远程 providers；若全部因缺少 API key 失败，会降级到 **FTS-only**（无 embedding / 无 vector 搜索）

---

## 2) 代码链路（从 memory_search 到向量召回）

下面按调用链路梳理“embedding 的功能在哪里被用起来”。

### 2.1 工具入口（Agent/Tool 层）

- openclaw-epeshared/src/agents/tools/memory-tool.ts
  - `memory_search`：调用 `getMemorySearchManager()` → `manager.search()`
  - `memory_get`：调用 `manager.readFile()`（只允许 MEMORY.md / memory/*.md 等白名单路径，避免任意文件读取）

默认插件注册：

- openclaw-epeshared/extensions/memory-core/index.ts
  - 将 `memory_search` / `memory_get` 注册到运行时，并注册 `openclaw memory ...` CLI

### 2.2 Manager 选择（QMD vs builtin）

- openclaw-epeshared/src/memory/search-manager.ts
  - 如果 `memory.backend = "qmd"`，优先创建 `QmdMemoryManager`
  - 若 QMD 不可用或运行失败，fallback 到内置 SQLite `MemoryIndexManager`

### 2.3 内置 SQLite manager（核心逻辑）

- openclaw-epeshared/src/memory/manager.ts
  - `MemoryIndexManager.search(query)`
    1)（可选）触发同步/索引更新
    2) `embedQueryWithTimeout(query)` 得到 query 向量
    3) 向量检索 `searchVector(...)`（sqlite-vec 或 JS fallback）
    4)（可选）BM25 关键词检索
    5)（可选）hybrid merge（向量+文本加权、MMR、多样性/时间衰减等）

- openclaw-epeshared/src/memory/manager-search.ts
  - `searchVector(...)`
    - sqlite-vec ready 时用 `vec_distance_cosine` 直接在 SQLite 里算距离
    - 否则 list 全 chunks，在 JS 里算 cosine similarity

### 2.4 索引构建（chunking + embed + 写入）

embedding 的“批处理/重试/缓存/写表”主要在：

- openclaw-epeshared/src/memory/manager-embedding-ops.ts
  - chunking：`chunkMarkdown(...)`
  - cache：`embedding_cache`（按 hash 命中）
  - embed：`embedBatchWithRetry(...)` / `embedQueryWithTimeout(...)`
  - remote batch（可选）：openai/gemini/voyage 的 batch runner

schema/扩展加载/vec 表维护在：

- openclaw-epeshared/src/memory/manager-sync-ops.ts
  - `ensureMemoryIndexSchema(...)`
  - `ensureVectorReady()` → `loadSqliteVecExtension()` → `CREATE VIRTUAL TABLE ... vec0`

---

## 3) 可选实现 A：QMD memory（外部工具链做 embedding + 检索）

当 `memory.backend = "qmd"` 时：

- OpenClaw 会使用 openclaw-epeshared/src/memory/qmd-manager.ts 来管理 QMD 索引更新与查询
- QMD 的路径集合、scope、更新/embedding 周期等由 openclaw-epeshared/src/memory/backend-config.ts 解析
- search-manager 会包一层 fallback：QMD query 报错时切换到 builtin SQLite（避免工具不可用）

从“embedding 的位置”角度：

- embedding 生成可能发生在 QMD 进程内部（或其配置的 pipeline）
- OpenClaw 在这里更多是 **驱动/调用**，而不是自己计算每个 chunk 的向量

---

## 4) 可选实现 B：LanceDB memory 插件（独立向量库）

插件实现位于：

- openclaw-epeshared/extensions/memory-lancedb/index.ts

特点：

- 使用 LanceDB 作为向量库，写入结构里直接存 `vector: number[]`
- embedding 生成使用 OpenAI `embeddings.create(...)`
- `vectorDimsForModel(...)` 在 config.ts 里固定了支持的 model→维度映射（目前只列了 `text-embedding-3-small/large`）

这条链路更像“传统向量数据库插件”：

- store/search 都在 LanceDB
- embedding provider 也在插件内部（不复用 src/memory/embeddings.ts 的多 provider 抽象）

---

## 5) 你在项目里看到 embedding 的“功能面”通常是什么？

在 OpenClaw 的语境里，embedding 主要服务于：

- **Semantic memory search / recall**：当 agent 需要回忆历史决定、偏好、TODO、人物等信息时，先 `memory_search` 获取相关片段，再用 `memory_get` 拉取必要上下文。
- **Hybrid retrieval**（可选）：向量“语义近似” + BM25 “关键词命中”混合，提升召回稳健性。
- **稳定性/降级**：当缺少 API key 或 embedding provider 不可用时，可以降级为 FTS-only（只做关键词检索）。

---

## 6) 小抄：调试/定位时常看的点

- embedding provider 是否可用：看 `createEmbeddingProvider(...)` 的选择与 fallback
- sqlite-vec 是否加载成功：看 `loadSqliteVecExtension(...)` + `ensureVectorReady()` 的状态
- 向量检索是否走 SQL：看 `searchVector(...)` 里是否进入 `vec_distance_cosine(...)` 分支
- 维度是否变化导致重建：`ensureVectorTable(dimensions)` 会在 dims 不一致时 drop/recreate

---

## 参考源码（建议从这几个文件开始读）

- openclaw-epeshared/src/memory/embeddings.ts
- openclaw-epeshared/src/memory/manager.ts
- openclaw-epeshared/src/memory/manager-search.ts
- openclaw-epeshared/src/memory/manager-embedding-ops.ts
- openclaw-epeshared/src/memory/manager-sync-ops.ts
- openclaw-epeshared/src/memory/memory-schema.ts
- openclaw-epeshared/src/agents/tools/memory-tool.ts
- openclaw-epeshared/extensions/memory-lancedb/index.ts
