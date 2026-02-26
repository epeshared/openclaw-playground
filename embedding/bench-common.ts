import { performance } from "node:perf_hooks";
import crypto from "node:crypto";
import { spawn } from "node:child_process";
import path from "node:path";

export type ParsedArgs = {
  _: string[];
  flags: Record<string, string | boolean>;
};

export function parseArgs(argv: string[]): ParsedArgs {
  const flags: Record<string, string | boolean> = {};
  const positional: string[] = [];
  for (let i = 0; i < argv.length; i += 1) {
    const cur = argv[i];
    if (!cur) continue;
    if (cur === "--") {
      positional.push(...argv.slice(i + 1));
      break;
    }
    if (!cur.startsWith("--")) {
      positional.push(cur);
      continue;
    }
    const eq = cur.indexOf("=");
    if (eq > 0) {
      const key = cur.slice(2, eq).trim();
      const value = cur.slice(eq + 1);
      if (key) flags[key] = value;
      continue;
    }
    const key = cur.slice(2).trim();
    if (!key) continue;
    const next = argv[i + 1];
    if (next && !next.startsWith("--")) {
      flags[key] = next;
      i += 1;
    } else {
      flags[key] = true;
    }
  }
  return { _: positional, flags };
}

export function readString(args: ParsedArgs, key: string, fallback?: string): string | undefined {
  const v = args.flags[key];
  if (typeof v === "string") {
    const trimmed = v.trim();
    return trimmed ? trimmed : fallback;
  }
  if (v === true) {
    return fallback;
  }
  return fallback;
}

export function readBool(args: ParsedArgs, key: string, fallback = false): boolean {
  const v = args.flags[key];
  if (v === true) return true;
  if (typeof v === "string") {
    const s = v.trim().toLowerCase();
    if (s === "1" || s === "true" || s === "yes" || s === "on") return true;
    if (s === "0" || s === "false" || s === "no" || s === "off") return false;
  }
  return fallback;
}

export function readInt(args: ParsedArgs, key: string, fallback: number): number {
  const raw = readString(args, key);
  if (!raw) return fallback;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) ? n : fallback;
}

export function readNumber(args: ParsedArgs, key: string, fallback: number): number {
  const raw = readString(args, key);
  if (!raw) return fallback;
  const n = Number(raw);
  return Number.isFinite(n) ? n : fallback;
}

export function clampInt(value: number, min: number, max: number): number {
  const v = Math.floor(value);
  return Math.max(min, Math.min(max, v));
}

export function nowMs(): number {
  return performance.now();
}

export function epochMs(): number {
  return Date.now();
}

export type Embedder = {
  id: string;
  model?: string;
  dims: number;
  embedBatch: (texts: string[]) => Promise<number[][]>;
  embedQuery: (text: string) => Promise<number[]>;
};

export function providerKeyForEmbedder(embedder: Embedder, apiKey?: string): string {
  // OpenClaw uses a providerKey to partition cache entries by credential.
  // For benchmarks we keep it stable but not sensitive.
  const raw = apiKey ? `key:${apiKey}` : "key:none";
  return crypto.createHash("sha256").update(`${embedder.id}:${embedder.model ?? ""}:${raw}`).digest("hex");
}

function fnv1a32(input: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < input.length; i += 1) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function xorshift32(seed: number): () => number {
  let x = seed >>> 0;
  return () => {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    return x >>> 0;
  };
}

function l2Normalize(vec: number[]): number[] {
  let sum = 0;
  for (const v of vec) sum += v * v;
  const inv = sum > 0 ? 1 / Math.sqrt(sum) : 0;
  return vec.map((v) => v * inv);
}

export function createFakeEmbedder(params?: {
  dims?: number;
  work?: number;
  normalize?: boolean;
}): Embedder {
  const dims = Math.max(1, Math.floor(params?.dims ?? 768));
  const work = Math.max(0, Math.floor(params?.work ?? 0));
  const normalize = Boolean(params?.normalize ?? true);

  const embedOne = (text: string): number[] => {
    // Deterministic per-text vectors so caching effects can be simulated.
    let seed = fnv1a32(text);
    // Optional extra CPU work to simulate expensive embeddings.
    for (let i = 0; i < work; i += 1) {
      seed = Math.imul(seed ^ 0x9e3779b9, 0x85ebca6b) >>> 0;
    }
    const rand = xorshift32(seed);
    const out: number[] = new Array(dims);
    for (let i = 0; i < dims; i += 1) {
      // Map uint32 -> [-1, 1)
      out[i] = (rand() / 0xffffffff) * 2 - 1;
    }
    return normalize ? l2Normalize(out) : out;
  };

  return {
    id: "fake",
    model: `fake-${dims}`,
    dims,
    embedBatch: async (texts) => texts.map((t) => embedOne(t)),
    embedQuery: async (text) => embedOne(text),
  };
}

export async function createRealEmbedder(params: {
  provider: "openai" | "gemini" | "voyage" | "mistral" | "local" | "auto";
  model?: string;
  apiKey?: string;
  baseUrl?: string;
  headersJson?: string;
  localModelPath?: string;
  localModelCacheDir?: string;
  agentDir?: string;
}): Promise<Embedder> {
  const embeddingsMod = await import(
    "../../openclaw-epeshared/src/memory/embeddings.ts"
  );
  const { createEmbeddingProvider } = embeddingsMod as typeof import("../../openclaw-epeshared/src/memory/embeddings.ts");

  const headers = (() => {
    if (!params.headersJson) return undefined;
    try {
      const parsed = JSON.parse(params.headersJson) as unknown;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, string>;
      }
      return undefined;
    } catch {
      return undefined;
    }
  })();

  const cfg = {
    // createEmbeddingProvider mostly uses the explicit remote/local params below.
    agents: { defaults: {} },
  };

  const providerResult = await createEmbeddingProvider({
    // Type intentionally loose (this is a benchmark harness).
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    config: cfg as any,
    agentDir: params.agentDir ?? "/tmp/openclaw-embed-bench-agent",
    provider: params.provider,
    remote:
      params.provider === "local"
        ? undefined
        : {
            apiKey: params.apiKey,
            baseUrl: params.baseUrl,
            headers,
            batch: {
              enabled: false,
              wait: true,
              concurrency: 1,
              pollIntervalMs: 2000,
              timeoutMinutes: 60,
            },
          },
    model: params.model ?? "",
    fallback: "none",
    local: {
      modelPath: params.localModelPath,
      modelCacheDir: params.localModelCacheDir,
    },
  });

  const provider = providerResult.provider;
  if (!provider) {
    throw new Error(providerResult.providerUnavailableReason ?? "embedding provider unavailable");
  }

  return {
    id: provider.id,
    model: provider.model,
    dims: provider.dims,
    embedBatch: async (texts) => await provider.embedBatch(texts),
    embedQuery: async (text) => await provider.embedQuery(text),
  };
}

export function randomText(params: { chars: number; seed?: string }): string {
  const chars = Math.max(0, Math.floor(params.chars));
  if (chars === 0) return "";
  const seed = params.seed ?? crypto.randomUUID();
  const prng = xorshift32(fnv1a32(seed));
  const alphabet =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:;!?-_/()[]{}\n";
  const out: string[] = new Array(chars);
  for (let i = 0; i < chars; i += 1) {
    out[i] = alphabet[prng() % alphabet.length] ?? " ";
  }
  return out.join("");
}

export type BatchResult = {
  batchSize: number;
  ms: number;
};

export class SampledLatency {
  private readonly cap: number;
  private readonly samples: number[] = [];
  private seen = 0;

  constructor(cap = 5000) {
    this.cap = Math.max(10, Math.floor(cap));
  }

  add(ms: number): void {
    if (!Number.isFinite(ms) || ms < 0) return;
    this.seen += 1;
    if (this.samples.length < this.cap) {
      this.samples.push(ms);
      return;
    }
    // Reservoir sampling.
    const j = Math.floor(Math.random() * this.seen);
    if (j < this.cap) {
      this.samples[j] = ms;
    }
  }

  summary(): { count: number; p50: number; p90: number; p95: number; p99: number } {
    const s = [...this.samples].sort((a, b) => a - b);
    const pick = (p: number) => {
      if (s.length === 0) return 0;
      const idx = Math.min(s.length - 1, Math.max(0, Math.floor(p * (s.length - 1))));
      return s[idx] ?? 0;
    };
    return {
      count: this.seen,
      p50: pick(0.5),
      p90: pick(0.9),
      p95: pick(0.95),
      p99: pick(0.99),
    };
  }
}

export async function runWithConcurrency<T>(
  tasks: Array<() => Promise<T>>,
  concurrency: number,
): Promise<T[]> {
  const limit = Math.max(1, Math.floor(concurrency));
  const results: T[] = new Array(tasks.length);
  let cursor = 0;

  const workers = new Array(limit).fill(null).map(async () => {
    while (true) {
      const i = cursor;
      cursor += 1;
      if (i >= tasks.length) return;
      results[i] = await tasks[i]!();
    }
  });
  await Promise.all(workers);
  return results;
}

export function splitIntoBatches<T>(items: T[], batchSize: number): T[][] {
  const size = Math.max(1, Math.floor(batchSize));
  const out: T[][] = [];
  for (let i = 0; i < items.length; i += size) {
    out.push(items.slice(i, i + size));
  }
  return out;
}

type BridgeRequest = { id: number; method: string; params: unknown };
type BridgeResponse =
  | { id: number; ok: true; result: unknown }
  | { id: number; ok: false; error: string };

export class SqliteBridge {
  private readonly proc;
  private nextId = 1;
  private readonly pending = new Map<
    number,
    { resolve: (v: unknown) => void; reject: (e: Error) => void }
  >();
  private readonly stderrChunks: string[] = [];

  constructor(params?: { python?: string; bridgePath?: string }) {
    const python = params?.python ?? process.env.PYTHON ?? "python";
    const bridgePath =
      params?.bridgePath ?? path.join(path.dirname(new URL(import.meta.url).pathname), "sqlite-bridge.py");
    this.proc = spawn(python, [bridgePath], {
      stdio: ["pipe", "pipe", "pipe"],
      env: process.env,
    });

    this.proc.stdout.setEncoding("utf-8");
    let buf = "";
    this.proc.stdout.on("data", (chunk: string) => {
      buf += chunk;
      while (true) {
        const idx = buf.indexOf("\n");
        if (idx < 0) break;
        const line = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 1);
        if (!line) continue;
        let resp: BridgeResponse;
        try {
          resp = JSON.parse(line) as BridgeResponse;
        } catch (e) {
          continue;
        }
        const entry = this.pending.get(resp.id);
        if (!entry) continue;
        this.pending.delete(resp.id);
        if (resp.ok) {
          entry.resolve(resp.result);
        } else {
          entry.reject(new Error(resp.error));
        }
      }
    });

    this.proc.stderr.setEncoding("utf-8");
    this.proc.stderr.on("data", (chunk: string) => {
      this.stderrChunks.push(chunk);
      if (this.stderrChunks.length > 50) this.stderrChunks.shift();
    });

    this.proc.on("exit", (code) => {
      const err = new Error(
        `sqlite bridge exited (code=${code}). stderr tail: ${this.stderrChunks.join("")}`,
      );
      for (const [id, p] of this.pending) {
        this.pending.delete(id);
        p.reject(err);
      }
    });
  }

  async request<T = unknown>(method: string, params: unknown): Promise<T> {
    const id = this.nextId++;
    const payload: BridgeRequest = { id, method, params };
    const line = JSON.stringify(payload) + "\n";
    const p = new Promise<unknown>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
    });
    this.proc.stdin.write(line, "utf-8");
    return (await p) as T;
  }

  async init(params: { dbPath: string; reset: boolean }): Promise<void> {
    await this.request("init", { dbPath: params.dbPath, reset: params.reset });
  }

  async cacheCheck(params: {
    dbPath: string;
    provider: string;
    model: string;
    providerKey: string;
    hashes: string[];
  }): Promise<Set<string>> {
    const res = (await this.request("cacheCheck", params)) as { hits: string[] };
    return new Set(res.hits ?? []);
  }

  async cachePut(params: {
    dbPath: string;
    provider: string;
    model: string;
    providerKey: string;
    updatedAtMs: number;
    items: Array<{ hash: string; embeddingJson: string }>;
  }): Promise<{ written: number }> {
    return (await this.request("cachePut", params)) as { written: number };
  }

  async chunksUpsert(params: {
    dbPath: string;
    sessionId: string;
    roundId: number;
    updatedAtMs: number;
    chunks: Array<{
      hash: string;
      startLine: number;
      endLine: number;
      text: string;
      embeddingHash: string;
    }>;
  }): Promise<{ written: number }> {
    return (await this.request("chunksUpsert", params)) as { written: number };
  }

  async chunksPruneMissing(params: {
    dbPath: string;
    sessionId: string;
    keepHashes: string[];
  }): Promise<{ deleted: number }> {
    return (await this.request("chunksPruneMissing", params)) as { deleted: number };
  }

  async close(): Promise<void> {
    try {
      await this.request("close", {});
    } catch {}
    try {
      this.proc.kill();
    } catch {}
  }
}

export function embeddingToJson(vec: number[]): string {
  // Match OpenClaw schema style (JSON string).
  return JSON.stringify(vec);
}

export async function embedMissingWithCache(params: {
  embedder: Embedder;
  bridge: SqliteBridge;
  dbPath: string;
  providerKey: string;
  chunkHashes: string[];
  chunkTextsByHash: Map<string, string>;
  batchSize: number;
  concurrency: number;
}): Promise<{ hits: number; misses: number; writtenCache: number; embedMs: number; dbMs: number }> {
  const tDb0 = nowMs();
  const hitsSet = await params.bridge.cacheCheck({
    dbPath: params.dbPath,
    provider: params.embedder.id,
    model: params.embedder.model ?? "",
    providerKey: params.providerKey,
    hashes: params.chunkHashes,
  });
  const dbCheckMs = nowMs() - tDb0;

  const misses: string[] = [];
  for (const h of params.chunkHashes) {
    if (!hitsSet.has(h)) misses.push(h);
  }

  const missTexts: string[] = [];
  for (const h of misses) {
    const text = params.chunkTextsByHash.get(h) ?? "";
    missTexts.push(text);
  }

  const missBatches = splitIntoBatches(
    misses.map((h, i) => ({ hash: h, text: missTexts[i] ?? "" })),
    params.batchSize,
  );

  const latchedEmbeddings: Array<{ hash: string; embeddingJson: string }> = [];
  const tEmb0 = nowMs();
  await runWithConcurrency(
    missBatches.map((batch) => async () => {
      const vectors = await params.embedder.embedBatch(batch.map((b) => b.text));
      for (let i = 0; i < batch.length; i += 1) {
        const entry = batch[i]!;
        const vec = vectors[i] ?? [];
        latchedEmbeddings.push({ hash: entry.hash, embeddingJson: embeddingToJson(vec) });
      }
    }),
    params.concurrency,
  );
  const embedMs = nowMs() - tEmb0;

  const tDb1 = nowMs();
  const putRes = await params.bridge.cachePut({
    dbPath: params.dbPath,
    provider: params.embedder.id,
    model: params.embedder.model ?? "",
    providerKey: params.providerKey,
    updatedAtMs: epochMs(),
    items: latchedEmbeddings,
  });
  const dbPutMs = nowMs() - tDb1;

  return {
    hits: params.chunkHashes.length - misses.length,
    misses: misses.length,
    writtenCache: putRes.written ?? 0,
    embedMs,
    dbMs: dbCheckMs + dbPutMs,
  };
}


export type SessionJsonlMessageRecord = {
  type: "message";
  message: {
    role: "user" | "assistant";
    content:
      | string
      | Array<{
          type: "text";
          text: string;
        }>;
  };
};

export function serializeSessionJsonl(messages: SessionJsonlMessageRecord[]): string {
  return messages.map((m) => JSON.stringify(m)).join("\n") + "\n";
}

function normalizeSessionText(value: string): string {
  return value.replace(/\s*\n+\s*/g, " ").replace(/\s+/g, " ").trim();
}

export function extractSessionText(content: unknown): string | null {
  if (typeof content === "string") {
    const normalized = normalizeSessionText(content);
    return normalized ? normalized : null;
  }
  if (!Array.isArray(content)) {
    return null;
  }
  const parts: string[] = [];
  for (const block of content) {
    if (!block || typeof block !== "object") {
      continue;
    }
    const record = block as { type?: unknown; text?: unknown };
    if (record.type !== "text" || typeof record.text !== "string") {
      continue;
    }
    const normalized = normalizeSessionText(record.text);
    if (normalized) {
      parts.push(normalized);
    }
  }
  if (parts.length === 0) {
    return null;
  }
  return parts.join(" ");
}

export type SessionEntryLike = {
  content: string;
  /** Maps each content line (0-indexed) to its 1-indexed JSONL source line. */
  lineMap: number[];
  totalJsonlLines: number;
  totalMessages: number;
};

export function buildSessionEntryFromJsonl(rawJsonl: string): SessionEntryLike {
  const lines = rawJsonl.split("\n");
  const collected: string[] = [];
  const lineMap: number[] = [];
  let totalMessages = 0;

  for (let jsonlIdx = 0; jsonlIdx < lines.length; jsonlIdx += 1) {
    const line = lines[jsonlIdx];
    if (!line || !line.trim()) {
      continue;
    }
    let record: unknown;
    try {
      record = JSON.parse(line);
    } catch {
      continue;
    }
    if (!record || typeof record !== "object" || (record as { type?: unknown }).type !== "message") {
      continue;
    }
    const message = (record as { message?: unknown }).message as
      | { role?: unknown; content?: unknown }
      | undefined;
    if (!message || typeof message.role !== "string") {
      continue;
    }
    if (message.role !== "user" && message.role !== "assistant") {
      continue;
    }
    const text = extractSessionText(message.content);
    if (!text) {
      continue;
    }
    totalMessages += 1;
    const label = message.role === "user" ? "User" : "Assistant";
    collected.push(`${label}: ${text}`);
    lineMap.push(jsonlIdx + 1);
  }

  return {
    content: collected.join("\n"),
    lineMap,
    totalJsonlLines: lines.length,
    totalMessages,
  };
}

export type MemoryChunkLike = {
  startLine: number;
  endLine: number;
  text: string;
  hash: string;
};

export function hashText(value: string): string {
  return crypto.createHash("sha256").update(value).digest("hex");
}

/**
 * Matches OpenClaw's `chunkMarkdown()` behavior (token budget approximated by chars).
 * See openclaw-epeshared/src/memory/internal.ts.
 */
export function chunkMarkdownLikeOpenClaw(
  content: string,
  chunking: { tokens: number; overlap: number },
): MemoryChunkLike[] {
  const lines = content.split("\n");
  if (lines.length === 0) {
    return [];
  }
  const maxChars = Math.max(32, Math.max(1, Math.floor(chunking.tokens)) * 4);
  const overlapChars = Math.max(0, Math.floor(chunking.overlap) * 4);
  const chunks: MemoryChunkLike[] = [];

  let current: Array<{ line: string; lineNo: number }> = [];
  let currentChars = 0;

  const flush = () => {
    if (current.length === 0) {
      return;
    }
    const firstEntry = current[0];
    const lastEntry = current[current.length - 1];
    if (!firstEntry || !lastEntry) {
      return;
    }
    const text = current.map((entry) => entry.line).join("\n");
    chunks.push({
      startLine: firstEntry.lineNo,
      endLine: lastEntry.lineNo,
      text,
      hash: hashText(text),
    });
  };

  const carryOverlap = () => {
    if (overlapChars <= 0 || current.length === 0) {
      current = [];
      currentChars = 0;
      return;
    }
    let acc = 0;
    const kept: Array<{ line: string; lineNo: number }> = [];
    for (let i = current.length - 1; i >= 0; i -= 1) {
      const entry = current[i];
      if (!entry) {
        continue;
      }
      acc += entry.line.length + 1;
      kept.unshift(entry);
      if (acc >= overlapChars) {
        break;
      }
    }
    current = kept;
    currentChars = kept.reduce((sum, entry) => sum + entry.line.length + 1, 0);
  };

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i] ?? "";
    const lineNo = i + 1;
    const segments: string[] = [];
    if (line.length === 0) {
      segments.push("");
    } else {
      for (let start = 0; start < line.length; start += maxChars) {
        segments.push(line.slice(start, start + maxChars));
      }
    }
    for (const segment of segments) {
      const lineSize = segment.length + 1;
      if (currentChars + lineSize > maxChars && current.length > 0) {
        flush();
        carryOverlap();
      }
      current.push({ line: segment, lineNo });
      currentChars += lineSize;
    }
  }
  flush();
  return chunks;
}

/**
 * Remap chunk start/end line numbers from flattened content lines to original JSONL lines.
 */
export function remapChunkLinesLikeOpenClaw(
  chunks: MemoryChunkLike[],
  lineMap: number[] | undefined,
): void {
  if (!lineMap || lineMap.length === 0) {
    return;
  }
  for (const chunk of chunks) {
    chunk.startLine = lineMap[chunk.startLine - 1] ?? chunk.startLine;
    chunk.endLine = lineMap[chunk.endLine - 1] ?? chunk.endLine;
  }
}

