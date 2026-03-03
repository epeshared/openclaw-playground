# bench-long-transcript: gateway + tui + benchmark

这个目录下的 `bench.ts` 是“长转录（long transcript）embedding”基准：它会生成一段很长的 transcript，按 OpenClaw 的 chunking 规则切块，然后调用 embedding provider + sqlite bridge 写库。

如果你想模拟一个更“真实”的现场（机器上有 OpenClaw Gateway 在跑，并且有一个客户端连着它），可以用本目录新增的 runner。

## 一键跑（脚本拉起 gateway，再跑 bench）

```bash
npx --yes tsx@4.21.0 openclaw-playground/embedding/bench-long-transcript/run-with-gateway.ts -- --provider=fake --mb=32
```

- `--` 后面的参数会原样传给 `bench.ts`。
- runner 默认会：
  - 用临时的 `OPENCLAW_STATE_DIR` 启动一个 Gateway（`gateway run --dev`）
    - 同时会把 `OPENCLAW_CONFIG_PATH` 指向这个临时目录里的 `openclaw.json`，避免自动读取你默认位置的 config（否则可能带着不同的 token 导致 TUI 认证失败）
  - 尝试启动一个“无交互”的 `openclaw tui --url ... --session main` 进程（best-effort，主要用于模拟“有个终端客户端连着”）
  - 执行 `bench.ts`
  - 结束后清理子进程

## 你想“真的开一个终端跑 TUI”（推荐）

由于 TUI 需要真实 TTY（交互终端），脚本里只能 best-effort 地 headless 启动它。
最贴近你描述的“两个终端”做法是：

终端 A（先启动 Gateway）：

```bash
cd /nvme5/xtang/openclaw-workspace/openclaw-epeshared
node ./openclaw.mjs gateway run --allow-unconfigured --force
```

终端 B（连接 TUI）：

```bash
cd /nvme5/xtang/openclaw-workspace/openclaw-epeshared
node ./openclaw.mjs tui --url ws://127.0.0.1:18789 --session main
```

终端 C（跑 bench）：

```bash
npx --yes tsx@4.21.0 openclaw-playground/embedding/bench-long-transcript/bench.ts --provider=fake --mb=32
```

## Runner 参数

```text
--url=<wsUrl>         默认 ws://127.0.0.1:18789
--port=<n>            默认 18789
--stateDir=<path>     让 gateway 用独立的 OPENCLAW_STATE_DIR（默认临时目录）
--spawnTui=<0|1>      是否尝试 headless 启动 tui（默认 1）
--tuiSession=<key>    默认 main
--gatewayDev=<0|1>    是否加 --dev（默认 1）
--keepAliveMs=<ms>    bench 结束后额外保持 gateway/tui 存活的时间（默认 0）
```
