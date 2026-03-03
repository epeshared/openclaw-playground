#!/usr/bin/env -S npx --yes tsx@4.21.0

import { spawn, type ChildProcess } from "node:child_process";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import path from "node:path";
import os from "node:os";
import fs from "node:fs";
import { performance } from "node:perf_hooks";

import { GatewayClient } from "../../../openclaw-epeshared/src/gateway/client.ts";

function usage(): string {
  return `Stress OpenClaw session transcript indexing (embedding bottleneck)

This orchestrates:
  1) (Optional) spawn an isolated OpenClaw Gateway
  2) create 1..N sessions ("rooms") via sessions.reset
  3) start 1..N WebSocket clients and spam chat.send (default) / chat.inject with long/highfreq messages
  4) (Optional) verify via HTTP /tools/invoke (memory_search)

Usage:
  npx --yes tsx@4.21.0 openclaw-playground/embedding/session-memory-stress/run-with-gateway.ts [options]

Gateway options:
  --spawnGateway=<0|1>          Spawn a new gateway (default: 1)
  --url=<wsUrl>                Gateway WS URL (default: ws://127.0.0.1:18789)
  --port=<n>                   Gateway port (default: 18789)
  --stateDir=<path>            OPENCLAW_STATE_DIR for spawned gateway (default: temp dir)
  --gatewayDev=<0|1>           Use 'gateway run --dev' (default: 1)
  --gatewayLogs=<0|1>          Print gateway logs to stderr (default: 1)
  --token=<token>              Gateway token (default: random)

Memory/session indexing config (written to OPENCLAW_CONFIG_PATH for spawned gateway):
  --embeddingProvider=<name>   memorySearch.provider (default: openai)
  --embeddingModel=<name>      memorySearch.model (default: provider default)
  --embeddingFallback=<name>   memorySearch.fallback (default: none)
  --remoteBaseUrl=<url>        memorySearch.remote.baseUrl (optional)
  --remoteApiKey=<key>         memorySearch.remote.apiKey (optional)
  --remoteBatch=<0|1>          memorySearch.remote.batch.enabled (default: 0)
  --remoteBatchWait=<0|1>      memorySearch.remote.batch.wait (default: 1)
  --remoteBatchConcurrency=<n> memorySearch.remote.batch.concurrency (default: 2)
  --vector=<0|1>               memorySearch.store.vector.enabled (default: 0)
  --cache=<0|1>                memorySearch.cache.enabled (default: 1)
  --chunkTokens=<n>            memorySearch.chunking.tokens (default: 400)
  --chunkOverlap=<n>           memorySearch.chunking.overlap (default: 80)
  --localModelPath=<path>      memorySearch.local.modelPath (optional)
  --localModelCacheDir=<path>  memorySearch.local.modelCacheDir (optional)
  --deltaBytes=<n>             session deltaBytes threshold (default: 1)
  --deltaMessages=<n>          session deltaMessages threshold (default: 1)

Load options:
  --chatMethod=<send|inject>   Which RPC to use for writes (default: send)
  --sendAbort=<0|1>            When chatMethod=send, immediately abort each run (default: 1)
  --sendTimeoutMs=<ms>         When chatMethod=send, pass timeout override (default: 2000; 0 disables)
  --clients=<n>                Number of WS clients (default: 1)
  --rooms=<n>                  Number of shared sessions (default: 1)
  --durationMs=<ms>            How long to send (default: 30000)
  --messageRate=<n>              Messages per second per client (default: 5)
  --messageChars=<n>           Approx chars per injected message (default: 8192)
  --inflight=<n>               Max concurrent in-flight inject RPCs per client (default: 8)

Verification (optional):
  --warmupSearch=<0|1>         Call memory_search once before sending (default: 1)
  --verifySearch=<0|1>         After sending, wait ~6s then memory_search for the token (default: 0)
  --verifyWaitMs=<ms>          Wait time before verifySearch (default: 6500)
  --verifyPollEveryMs=<ms>     If >0, keep polling until first hit (default: 0)
  --verifyTimeoutMs=<ms>       Max time to wait for first hit (default: 120000)

Stats:
  --statsEveryMs=<ms>          Print periodic stats (default: 2000; 0 disables)

Indexing lag probe (optional):
  --probeSearchEveryMs=<ms>    Periodically call memory_search(marker) during the run (default: 0)
  --probeMaxResults=<n>        maxResults for probe search (default: 1)

Notes:
  - Session transcript indexing is debounced (~5s) and async; verification waits for that.
  - To make embeddings the bottleneck, switch --embeddingProvider=local or a remote provider.
`;
}

type Parsed = {
  flags: Record<string, string | boolean>;
};

function parseArgs(argv: string[]): Parsed {
  const flags: Record<string, string | boolean> = {};
  for (const raw of argv) {
    if (!raw.startsWith("--")) continue;
    const eq = raw.indexOf("=");
    if (eq === -1) {
      flags[raw.slice(2)] = true;
    } else {
      flags[raw.slice(2, eq)] = raw.slice(eq + 1);
    }
  }
  return { flags };
}

function readString(flags: Record<string, string | boolean>, key: string, def?: string): string | undefined {
  const v = flags[key];
  if (typeof v === "string") return v;
  if (v === true) return "true";
  return def;
}

function readNonEmptyString(
  flags: Record<string, string | boolean>,
  key: string,
  def?: string,
): string | undefined {
  const v = readString(flags, key, def);
  const s = v?.trim() ?? "";
  return s ? s : undefined;
}

function readInt(flags: Record<string, string | boolean>, key: string, def: number): number {
  const v = readString(flags, key);
  if (!v) return def;
  const n = Number.parseInt(v, 10);
  return Number.isFinite(n) ? n : def;
}

function readBool(flags: Record<string, string | boolean>, key: string, def: boolean): boolean {
  const v = flags[key];
  if (typeof v === "boolean") return v;
  if (typeof v === "string") {
    const s = v.trim().toLowerCase();
    if (s === "1" || s === "true" || s === "yes" || s === "y") return true;
    if (s === "0" || s === "false" || s === "no" || s === "n") return false;
  }
  return def;
}

type Histogram = {
  buckets: number[];
  counts: number[];
  count: number;
  sum: number;
  min: number;
  max: number;
};

function createHistogram(): Histogram {
  const buckets = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000];
  return {
    buckets,
    counts: new Array(buckets.length + 1).fill(0),
    count: 0,
    sum: 0,
    min: Number.POSITIVE_INFINITY,
    max: 0,
  };
}

function recordHistogram(h: Histogram, valueMs: number): void {
  const v = Math.max(0, valueMs);
  h.count += 1;
  h.sum += v;
  h.min = Math.min(h.min, v);
  h.max = Math.max(h.max, v);
  let idx = h.buckets.findIndex((b) => v <= b);
  if (idx < 0) idx = h.counts.length - 1;
  h.counts[idx] += 1;
}

function approxPercentileMs(h: Histogram, p: number): number {
  if (h.count <= 0) return 0;
  const target = Math.ceil(Math.min(1, Math.max(0, p)) * h.count);
  let seen = 0;
  for (let i = 0; i < h.counts.length; i += 1) {
    seen += h.counts[i] ?? 0;
    if (seen >= target) {
      return h.buckets[i] ?? (h.buckets.at(-1)! + 1);
    }
  }
  return h.buckets.at(-1)! + 1;
}

function fmtMs(n: number): string {
  if (!Number.isFinite(n)) return "?ms";
  if (n < 1000) return `${Math.round(n)}ms`;
  return `${(n / 1000).toFixed(2)}s`;
}

async function sleep(ms: number): Promise<void> {
  await new Promise((r) => setTimeout(r, ms));
}

function spawnLogged(
  label: string,
  command: string,
  args: string[],
  opts: {
    cwd: string;
    env?: NodeJS.ProcessEnv;
    stdio?: ("pipe" | "ignore" | "inherit")[];
  },
): ChildProcess {
  const child = spawn(command, args, {
    cwd: opts.cwd,
    env: opts.env,
    stdio: opts.stdio ?? ["ignore", "pipe", "pipe"],
  });

  if (child.stdout) {
    child.stdout.on("data", (chunk) => {
      process.stderr.write(`[${label}] ${String(chunk)}`);
    });
  }
  if (child.stderr) {
    child.stderr.on("data", (chunk) => {
      process.stderr.write(`[${label}] ${String(chunk)}`);
    });
  }
  child.on("exit", (code, signal) => {
    process.stderr.write(`[${label}] exited: code=${code ?? "null"} signal=${signal ?? "null"}\n`);
  });

  return child;
}

async function waitForGatewayHealth(params: {
  openclawRepoDir: string;
  env: NodeJS.ProcessEnv;
  url: string;
  token: string;
  retries: number;
  delayMs: number;
}): Promise<void> {
  for (let i = 1; i <= params.retries; i += 1) {
    const child = spawn(
      process.execPath,
      ["./openclaw.mjs", "gateway", "health", "--url", params.url, "--token", params.token],
      {
      cwd: params.openclawRepoDir,
      env: params.env,
      stdio: ["ignore", "ignore", "ignore"],
      },
    );

    const ok: boolean = await new Promise((resolve) => {
      child.on("exit", (code) => resolve(code === 0));
      child.on("error", () => resolve(false));
    });

    if (ok) {
      return;
    }
    await sleep(params.delayMs);
  }
  throw new Error("Gateway did not become healthy in time");
}

async function terminate(proc: ChildProcess | null | undefined): Promise<void> {
  if (!proc || proc.killed) return;
  const pid = proc.pid;
  if (!pid) return;

  proc.kill("SIGINT");
  const exited = await Promise.race([
    new Promise<boolean>((resolve) => proc.once("exit", () => resolve(true))),
    sleep(1500).then(() => false),
  ]);
  if (exited) return;

  proc.kill("SIGTERM");
  const exited2 = await Promise.race([
    new Promise<boolean>((resolve) => proc.once("exit", () => resolve(true))),
    sleep(1500).then(() => false),
  ]);
  if (exited2) return;

  proc.kill("SIGKILL");
  await Promise.race([
    new Promise<boolean>((resolve) => proc.once("exit", () => resolve(true))),
    sleep(1500).then(() => false),
  ]);

  process.stderr.write(`[stress] forced kill: gateway pid=${pid}\n`);
}

function wsToHttpUrl(wsUrl: string): string {
  const u = new URL(wsUrl);
  u.protocol = u.protocol === "wss:" ? "https:" : "http:";
  u.pathname = "/tools/invoke";
  u.search = "";
  u.hash = "";
  return u.toString();
}

async function toolsInvoke(params: {
  httpUrl: string;
  token: string;
  sessionKey: string;
  tool: string;
  args: Record<string, unknown>;
}): Promise<unknown> {
  const res = await fetch(params.httpUrl, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${params.token}`,
    },
    body: JSON.stringify({
      tool: params.tool,
      args: params.args,
      sessionKey: params.sessionKey,
    }),
  });

  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!res.ok) {
    throw new Error(`tools.invoke failed: HTTP ${res.status} ${typeof body === "string" ? body : JSON.stringify(body)}`);
  }
  return body;
}

function extractMemorySearchResults(payload: unknown): unknown[] {
  if (!payload || typeof payload !== "object") return [];
  const obj = payload as Record<string, unknown>;
  const details = obj.details;
  if (details && typeof details === "object") {
    const d = details as Record<string, unknown>;
    if (Array.isArray(d.results)) return d.results;
  }
  const result = obj.result;
  if (result && typeof result === "object") {
    const r = result as Record<string, unknown>;
    // HTTP /tools/invoke returns { ok: true, result: { content, details } }
    // so the actual tool payload is typically under result.details.
    const rDetails = r.details;
    if (rDetails && typeof rDetails === "object") {
      const rd = rDetails as Record<string, unknown>;
      if (Array.isArray(rd.results)) return rd.results;
    }
    if (Array.isArray(r.results)) return r.results;
  }
  if (Array.isArray(obj.results)) return obj.results;
  const content = obj.content;
  if (Array.isArray(content) && content.length > 0) {
    const first = content[0];
    if (first && typeof first === "object") {
      const text = (first as Record<string, unknown>).text;
      if (typeof text === "string" && text.trim().startsWith("{")) {
        try {
          const parsed = JSON.parse(text) as unknown;
          return extractMemorySearchResults(parsed);
        } catch {
          // ignore
        }
      }
    }
  }
  return [];
}

function isMemorySearchHit(payload: unknown, marker: string): boolean {
  const trimmed = marker.trim();
  if (!trimmed) return false;

  const results = extractMemorySearchResults(payload);
  if (results.length > 0) {
    // Best-effort: look for the marker in common fields.
    for (const item of results) {
      if (!item || typeof item !== "object") continue;
      const obj = item as Record<string, unknown>;
      const snippet = typeof obj.snippet === "string" ? obj.snippet : "";
      const citation = typeof obj.citation === "string" ? obj.citation : "";
      const path = typeof obj.path === "string" ? obj.path : "";
      if (snippet.includes(trimmed)) return true;
      if (citation.includes(trimmed)) return true;
      if (path.includes("sessions/") && snippet) return true;
    }
    return true;
  }

  // Fallback: some tool envelopes don’t expose `results` cleanly.
  // Avoid false positives by requiring both marker + a sessions/snippet signal.
  let serialized = "";
  try {
    serialized = JSON.stringify(payload);
  } catch {
    serialized = String(payload);
  }
  if (!serialized.includes(trimmed)) return false;
  const looksLikeSessionSnippet =
    serialized.includes("\"snippet\"") ||
    serialized.includes("Source: sessions/") ||
    serialized.includes("\"citation\"") ||
    serialized.includes("sessions/");
  return looksLikeSessionSnippet;
}

function buildMessage(params: {
  marker: string;
  clientId: number;
  seq: number;
  messageChars: number;
}): string {
  const header = `${params.marker} client=${params.clientId} seq=${params.seq} `;
  const filler = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua ";
  if (header.length >= params.messageChars) {
    return header.slice(0, params.messageChars);
  }
  const need = params.messageChars - header.length;
  const reps = Math.max(1, Math.ceil(need / filler.length));
  return header + filler.repeat(reps).slice(0, need);
}

async function connectClient(params: {
  url: string;
  token: string;
  label: string;
  timeoutMs: number;
}): Promise<GatewayClient> {
  return await new Promise<GatewayClient>((resolve, reject) => {
    let client: GatewayClient | null = null;
    const timer = setTimeout(() => {
      try {
        client?.stop();
      } catch {
        // ignore
      }
      reject(
        new Error(`gateway client connect timeout after ${params.timeoutMs}ms (${params.label})`),
      );
    }, Math.max(1, params.timeoutMs));

    client = new GatewayClient({
      url: params.url,
      token: params.token,
      clientDisplayName: params.label,
      onHelloOk: () => {
        clearTimeout(timer);
        resolve(client!);
      },
      onConnectError: (err) => {
        clearTimeout(timer);
        try {
          client?.stop();
        } catch {
          // ignore
        }
        reject(err);
      },
    });
    client.start();
  });
}

async function main(): Promise<void> {
  const parsed = parseArgs(process.argv.slice(2));
  if (parsed.flags.help) {
    console.log(usage());
    return;
  }

  const here = path.dirname(fileURLToPath(import.meta.url));
  const openclawRepoDir = path.resolve(here, "../../../openclaw-epeshared");

  const spawnGateway = readBool(parsed.flags, "spawnGateway", true);
  const url = readString(parsed.flags, "url", "ws://127.0.0.1:18789")!;
  const port = readInt(parsed.flags, "port", 18789);
  const gatewayDev = readBool(parsed.flags, "gatewayDev", true);
  const gatewayLogs = readBool(parsed.flags, "gatewayLogs", true);
  const embeddingProvider = readString(parsed.flags, "embeddingProvider", "openai")!;
  const embeddingModel = readNonEmptyString(parsed.flags, "embeddingModel");
  const embeddingFallback = readString(parsed.flags, "embeddingFallback", "none")!;
  const remoteBaseUrl = readNonEmptyString(parsed.flags, "remoteBaseUrl");
  const remoteApiKey = readNonEmptyString(parsed.flags, "remoteApiKey");
  const remoteBatch = readBool(parsed.flags, "remoteBatch", false);
  const remoteBatchWait = readBool(parsed.flags, "remoteBatchWait", true);
  const remoteBatchConcurrency = Math.max(1, readInt(parsed.flags, "remoteBatchConcurrency", 2));
  const vectorEnabled = readBool(parsed.flags, "vector", false);
  const cacheEnabled = readBool(parsed.flags, "cache", true);
  const chunkTokens = Math.max(32, readInt(parsed.flags, "chunkTokens", 400));
  const chunkOverlap = Math.max(0, readInt(parsed.flags, "chunkOverlap", 80));
  const localModelPath = readNonEmptyString(parsed.flags, "localModelPath");
  const localModelCacheDir = readNonEmptyString(parsed.flags, "localModelCacheDir");
  const deltaBytes = readInt(parsed.flags, "deltaBytes", 1);
  const deltaMessages = readInt(parsed.flags, "deltaMessages", 1);

  const chatMethodRaw = readString(parsed.flags, "chatMethod", "send")!;
  const chatMethod = chatMethodRaw.trim().toLowerCase();
  if (chatMethod !== "send" && chatMethod !== "inject") {
    throw new Error(`invalid --chatMethod=${chatMethodRaw} (expected send|inject)`);
  }
  const sendAbort = readBool(parsed.flags, "sendAbort", true);
  const sendTimeoutMsRaw = readInt(parsed.flags, "sendTimeoutMs", 2000);
  const sendTimeoutMs = sendTimeoutMsRaw > 0 ? sendTimeoutMsRaw : undefined;

  const clients = Math.max(1, readInt(parsed.flags, "clients", 1));
  const rooms = Math.max(1, readInt(parsed.flags, "rooms", 1));
  const durationMs = Math.max(1, readInt(parsed.flags, "durationMs", 30_000));
  const messageRate = Math.max(0, readInt(parsed.flags, "messageRate", 5));
  const messageChars = Math.max(1, readInt(parsed.flags, "messageChars", 8192));
  const inflight = Math.max(1, readInt(parsed.flags, "inflight", 8));
  const warmupSearch = readBool(parsed.flags, "warmupSearch", true);
  const verifySearch = readBool(parsed.flags, "verifySearch", false);
  const verifyWaitMs = Math.max(0, readInt(parsed.flags, "verifyWaitMs", 6500));
  const verifyPollEveryMs = Math.max(0, readInt(parsed.flags, "verifyPollEveryMs", 0));
  const verifyTimeoutMs = Math.max(1, readInt(parsed.flags, "verifyTimeoutMs", 120_000));
  const statsEveryMs = Math.max(0, readInt(parsed.flags, "statsEveryMs", 2000));
  const probeSearchEveryMs = Math.max(0, readInt(parsed.flags, "probeSearchEveryMs", 0));
  const probeMaxResults = Math.max(1, readInt(parsed.flags, "probeMaxResults", 1));

  const stateDir =
    readString(parsed.flags, "stateDir") ??
    path.join(os.tmpdir(), `openclaw-session-memory-stress-${process.pid}`);
  const configPath = path.join(stateDir, "openclaw.json");

  const token =
    (readString(parsed.flags, "token") && readString(parsed.flags, "token")!.trim())
      ? readString(parsed.flags, "token")!.trim()
      : crypto.randomUUID();

  const stressRunId = crypto.randomUUID();
  const stressMarker = `stresstoken${stressRunId.replaceAll("-", "")}`;
  const httpUrl = wsToHttpUrl(url);

  const env: NodeJS.ProcessEnv = {
    ...process.env,
    OPENCLAW_STATE_DIR: stateDir,
    OPENCLAW_CONFIG_PATH: configPath,
    OPENCLAW_GATEWAY_TOKEN: token,
  };

  let gatewayProc: ChildProcess | undefined;
  const activeClients: GatewayClient[] = [];

  const cleanup = async () => {
    for (const c of activeClients) {
      try {
        c.stop();
      } catch {
        // ignore
      }
    }
    activeClients.length = 0;
    await terminate(gatewayProc);
  };

  process.on("SIGINT", () => {
    void cleanup().finally(() => process.exit(130));
  });
  process.on("SIGTERM", () => {
    void cleanup().finally(() => process.exit(143));
  });

  try {
    await fs.promises.mkdir(stateDir, { recursive: true });

    if (spawnGateway) {
      const config = {
        plugins: {
          slots: {
            memory: "memory-core",
          },
        },
        agents: {
          defaults: {
            memorySearch: {
              provider: embeddingProvider,
              model: embeddingModel,
              // If provider is missing API keys / deps, OpenClaw can degrade to FTS-only.
              fallback: embeddingFallback,
              experimental: { sessionMemory: true },
              sources: ["sessions"],
              local: {
                modelPath: localModelPath,
                modelCacheDir: localModelCacheDir,
              },
              cache: {
                enabled: cacheEnabled,
              },
              chunking: {
                tokens: chunkTokens,
                overlap: chunkOverlap,
              },
              remote: {
                baseUrl: remoteBaseUrl,
                apiKey: remoteApiKey,
                batch: {
                  enabled: remoteBatch,
                  wait: remoteBatchWait,
                  concurrency: remoteBatchConcurrency,
                },
              },
              store: {
                vector: { enabled: vectorEnabled },
              },
              query: {
                hybrid: { enabled: true },
              },
              sync: {
                sessions: {
                  deltaBytes,
                  deltaMessages,
                },
              },
            },
          },
        },
      };
      await fs.promises.writeFile(configPath, JSON.stringify(config, null, 2), "utf-8");

      const gatewayArgs = [
        "./openclaw.mjs",
        "gateway",
        "run",
        "--allow-unconfigured",
        "--force",
        "--ws-log",
        "compact",
        "--port",
        String(port),
        "--token",
        token,
      ];
      if (gatewayDev) {
        gatewayArgs.splice(3, 0, "--dev");
      }

      gatewayProc = spawnLogged("gateway", process.execPath, gatewayArgs, {
        cwd: openclawRepoDir,
        env,
        stdio: gatewayLogs ? ["ignore", "pipe", "pipe"] : ["ignore", "ignore", "ignore"],
      });

      await waitForGatewayHealth({
        openclawRepoDir,
        env,
        url,
        token,
        retries: 60,
        delayMs: 250,
      });
    }

    // Connect WS clients (simulating multiple TUIs).
    for (let i = 0; i < clients; i += 1) {
      const c = await connectClient({
        url,
        token,
        label: `stress-client-${i}`,
        timeoutMs: 10_000,
      });
      activeClients.push(c);
    }

    const roomKeys = Array.from({ length: rooms }, (_, idx) => `stress:group:room-${idx + 1}`);
    // Ensure sessions exist.
    await Promise.all(
      roomKeys.map(async (key) => {
        await activeClients[0]!.request("sessions.reset", { key, reason: "new" });
      }),
    );

    if (warmupSearch) {
      try {
        await toolsInvoke({
          httpUrl,
          token,
          sessionKey: roomKeys[0]!,
          tool: "memory_search",
          args: { query: "warmup", maxResults: 1 },
        });
      } catch (err) {
        process.stderr.write(`[stress] warmup memory_search failed: ${String(err)}\n`);
      }
    }

    const start = Date.now();
    const injectLatency = createHistogram();
    const probeLatency = createHistogram();
    const endAt = start + durationMs;
    const intervalMs = messageRate <= 0 ? null : Math.max(0, Math.floor(1000 / messageRate));

    let sent = 0;
    let errors = 0;

    let firstProbeHitMs: number | null = null;
    let probeOk = 0;
    let probeErrors = 0;
    let probeLastCount = 0;

    let statsTimer: NodeJS.Timeout | null = null;
    if (statsEveryMs > 0) {
      statsTimer = setInterval(() => {
        const elapsedMs = Date.now() - start;
        const qpsNow = elapsedMs > 0 ? sent / (elapsedMs / 1000) : 0;
        const p50 = approxPercentileMs(injectLatency, 0.5);
        const p95 = approxPercentileMs(injectLatency, 0.95);
        const p99 = approxPercentileMs(injectLatency, 0.99);
        const avg = injectLatency.count > 0 ? injectLatency.sum / injectLatency.count : 0;
        const base =
          `[stress] stats: elapsed=${fmtMs(elapsedMs)} sent=${sent} errors=${errors} qps=${qpsNow.toFixed(2)} ` +
          `injectLatency(p50/p95/p99/avg)=${fmtMs(p50)}/${fmtMs(p95)}/${fmtMs(p99)}/${fmtMs(avg)} inflight=${inflight}`;
        if (probeSearchEveryMs > 0) {
          const sp95 = approxPercentileMs(probeLatency, 0.95);
          const hit = firstProbeHitMs != null ? ` firstHit=${fmtMs(firstProbeHitMs)}` : " firstHit=?";
          process.stderr.write(
            `${base} probe(ok/err/last/p95)=${probeOk}/${probeErrors}/${probeLastCount}/${fmtMs(sp95)}${hit}\n`,
          );
        } else {
          process.stderr.write(`${base}\n`);
        }
      }, statsEveryMs);
      statsTimer.unref?.();
    }

    let probeTimer: NodeJS.Timeout | null = null;
    if (probeSearchEveryMs > 0) {
      probeTimer = setInterval(() => {
        void (async () => {
          const t0 = performance.now();
          try {
            const res = await toolsInvoke({
              httpUrl,
              token,
              sessionKey: roomKeys[0]!,
              tool: "memory_search",
              args: { query: stressMarker, maxResults: probeMaxResults },
            });
            const results = extractMemorySearchResults(res);
            probeLastCount = results.length;
            probeOk += 1;
            recordHistogram(probeLatency, performance.now() - t0);
            if (firstProbeHitMs == null && isMemorySearchHit(res, stressMarker)) {
              firstProbeHitMs = Date.now() - start;
            }
          } catch (err) {
            probeErrors += 1;
            process.stderr.write(`[stress] probe memory_search failed: ${String(err)}\n`);
            recordHistogram(probeLatency, performance.now() - t0);
          }
        })();
      }, probeSearchEveryMs);
      probeTimer.unref?.();
    }

    const perClientLoops = activeClients.map(async (client, clientId) => {
      let seq = 0;
      const inFlight = new Set<Promise<void>>();
      while (Date.now() < endAt) {
        if (intervalMs != null && intervalMs > 0) {
          await sleep(intervalMs);
        }

        const roomKey = roomKeys[Math.floor(Math.random() * roomKeys.length)]!;
        const message = buildMessage({
          marker: stressMarker,
          clientId,
          seq,
          messageChars,
        });
        const idempotencyKey = `${stressRunId}:${clientId}:${seq}`;
        seq += 1;

        const op = (async () => {
          const t0 = performance.now();
          try {
            if (chatMethod === "inject") {
              await client.request("chat.inject", {
                sessionKey: roomKey,
                message,
                label: `c${clientId}`,
              });
            } else {
              await client.request("chat.send", {
                sessionKey: roomKey,
                message,
                idempotencyKey,
                timeoutMs: sendTimeoutMs,
              });
              if (sendAbort) {
                await client.request("chat.abort", {
                  sessionKey: roomKey,
                  runId: idempotencyKey,
                });
              }
            }
            sent += 1;
          } catch (err) {
            errors += 1;
            process.stderr.write(
              `[stress] chat.${chatMethod} failed: ${String(err)}\n`,
            );
          } finally {
            recordHistogram(injectLatency, performance.now() - t0);
          }
        })();

        inFlight.add(op);
        op.finally(() => inFlight.delete(op)).catch(() => {
          // ignore
        });

        if (inFlight.size >= inflight) {
          await Promise.race(inFlight);
        }
      }
      await Promise.allSettled(Array.from(inFlight));
    });

    await Promise.all(perClientLoops);

    if (statsTimer) {
      clearInterval(statsTimer);
      statsTimer = null;
    }
    if (probeTimer) {
      clearInterval(probeTimer);
      probeTimer = null;
    }

    const elapsedMs = Date.now() - start;
    process.stderr.write(
      `[stress] done: clients=${clients} rooms=${rooms} sent=${sent} errors=${errors} elapsedMs=${elapsedMs} runId=${stressRunId} marker=${stressMarker}\n`,
    );

    if (verifySearch) {
      // Wait for session transcript indexing debounce + async sync.
      await sleep(verifyWaitMs);

      const verifyOnce = async () => {
        const res = await toolsInvoke({
          httpUrl,
          token,
          sessionKey: roomKeys[0]!,
          tool: "memory_search",
          args: { query: stressMarker, maxResults: 5 },
        });
        const results = extractMemorySearchResults(res);
        const hit = isMemorySearchHit(res, stressMarker);
        const summary =
          res && typeof res === "object" && "result" in (res as Record<string, unknown>)
            ? (res as Record<string, unknown>).result
            : res;
        return { res, results, hit, summary };
      };

      const tVerifyStart = Date.now();
      if (verifyPollEveryMs > 0) {
        while (Date.now() - tVerifyStart < verifyTimeoutMs) {
          const out = await verifyOnce();
          if (out.hit) {
            process.stderr.write(
              `[stress] verify hit after ${fmtMs(Date.now() - tVerifyStart)} results=${out.results.length}\n`,
            );
            process.stderr.write(
              `[stress] memory_search response: ${JSON.stringify(out.summary).slice(0, 4000)}\n`,
            );
            break;
          }
          await sleep(verifyPollEveryMs);
        }
        if (Date.now() - tVerifyStart >= verifyTimeoutMs) {
          const out = await verifyOnce();
          process.stderr.write(
            `[stress] verify timed out after ${fmtMs(verifyTimeoutMs)} results=${out.results.length}\n`,
          );
          process.stderr.write(
            `[stress] memory_search response: ${JSON.stringify(out.summary).slice(0, 4000)}\n`,
          );
        }
      } else {
        const out = await verifyOnce();
        process.stderr.write(`[stress] memory_search response: ${JSON.stringify(out.summary).slice(0, 4000)}\n`);
      }
    }
  } finally {
    await cleanup();
  }
}

main().catch((err) => {
  process.stderr.write(`[stress] fatal: ${String(err)}\n`);
  process.exit(1);
});
