#!/usr/bin/env -S npx --yes tsx@4.21.0
import {
  buildSessionEntryFromJsonl,
  chunkMarkdownLikeOpenClaw,
  clampInt,
  createFakeEmbedder,
  createRealEmbedder,
  epochMs,
  nowMs,
  parseArgs,
  providerKeyForEmbedder,
  readBool,
  readInt,
  readString,
  remapChunkLinesLikeOpenClaw,
  runWithConcurrency,
  SqliteBridge,
  SampledLatency,
  serializeSessionJsonl,
  splitIntoBatches,
  type Embedder,
  type SessionJsonlMessageRecord,
  embedMissingWithCache,
  randomText,
} from "../bench-common.ts";

function usage(): string {
  return `High-frequency chat embedding benchmark (OpenClaw-like indexing chain)

Generates a session transcript JSONL, extracts session text like OpenClaw,
chunks it using OpenClaw's chunkMarkdown heuristic, then embeds the chunks.

Usage:
  npx --yes tsx@4.21.0 openclaw-playground/embedding/bench-highfreq-chat/bench.ts [options]

Options:
  --provider=fake|openai|gemini|voyage|mistral|local|auto   (default: fake)
  --model=<id>                                             (default: provider default)
  --apiKey=<key> / --apiKeyEnv=<ENV>                       (real providers)
  --baseUrl=<url>                                          (real providers, optional)
  --headersJson='{"X-Foo":"bar"}'                       (real providers, optional)

  --seconds=<n>        total run time in seconds (default: 20). Also acts as tick count.
  --mps=<n>            transcript messages per second / per tick (default: 20)
  --chars=<n>          characters per message (default: 200)

  --deltaMessages=<n>  session-delta threshold (default: 50)
  --deltaBytes=<n>     session-delta threshold (default: 100000)

  --chunkTokens=<n>    OpenClaw chunking tokens (default: 400)
  --chunkOverlap=<n>   OpenClaw chunk overlap tokens (default: 80)

  --batchSize=<n>      embedBatch size (default: 8)
  --concurrency=<n>    parallel batch requests (default: 2)
  --warmupBatches=<n>  warmup embed batches (default: 2)

  --dbPath=<path>      sqlite db path (default: /tmp/openclaw-bench-highfreq.sqlite)
  --resetDb=<0|1>      reset db at start (default: 1)
  --pruneMissing=<0|1> delete chunks missing after sync (default: 0)

  --fakeDims=<n>       fake embed dims (default: 768)
  --fakeWork=<n>       fake extra CPU work (default: 0)

Output:
  Prints a JSON summary to stdout.
`;
}

async function makeEmbedder(args: ReturnType<typeof parseArgs>): Promise<{ embedder: Embedder; apiKey?: string }> {
  const provider =
    (readString(args, "provider", "fake") as
      | "fake"
      | "openai"
      | "gemini"
      | "voyage"
      | "mistral"
      | "local"
      | "auto") ?? "fake";
  if (provider === "fake") {
    return {
      embedder: createFakeEmbedder({
      dims: readInt(args, "fakeDims", 768),
      work: readInt(args, "fakeWork", 0),
      normalize: true,
      }),
      apiKey: undefined,
    };
  }
  const apiKeyEnv = readString(args, "apiKeyEnv");
  const apiKey = readString(args, "apiKey") ?? (apiKeyEnv ? process.env[apiKeyEnv] : undefined);
  const embedder = await createRealEmbedder({
      provider,
      model: readString(args, "model"),
      apiKey,
      baseUrl: readString(args, "baseUrl"),
      headersJson: readString(args, "headersJson"),
    });
  return { embedder, apiKey };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.flags.help) {
    console.log(usage());
    return;
  }

  const seconds = Math.max(1, readInt(args, "seconds", 20));
  const mps = Math.max(1, readInt(args, "mps", 20));
  const chars = Math.max(1, readInt(args, "chars", 200));
  const deltaMessages = clampInt(readInt(args, "deltaMessages", 50), 0, 1_000_000_000);
  const deltaBytes = clampInt(readInt(args, "deltaBytes", 100_000), 0, 2_000_000_000);
  const chunkTokens = clampInt(readInt(args, "chunkTokens", 400), 32, 100_000);
  const chunkOverlap = clampInt(readInt(args, "chunkOverlap", 80), 0, Math.max(0, chunkTokens - 1));
  const batchSize = clampInt(readInt(args, "batchSize", 8), 1, 2048);
  const concurrency = clampInt(readInt(args, "concurrency", 2), 1, 64);
  const warmupBatches = clampInt(readInt(args, "warmupBatches", 2), 0, 50);
  const dbPath = readString(args, "dbPath", "/tmp/openclaw-bench-highfreq.sqlite")!;
  const resetDb = readBool(args, "resetDb", true);
  const pruneMissing = readBool(args, "pruneMissing", false);

  const { embedder, apiKey } = await makeEmbedder(args);
  const providerKey = providerKeyForEmbedder(embedder, apiKey);

  const bridge = new SqliteBridge();
  await bridge.init({ dbPath, reset: resetDb });

  const sessionId = "bench:highfreq";
  const messages: SessionJsonlMessageRecord[] = [];
  let msgIdx = 0;

  const syncRuns: Array<{
    tick: number;
    appendedMessages: number;
    appendedBytes: number;
    sessionMessages: number;
    chunks: number;
    uniqueChunkHashes: number;
    cacheHits: number;
    cacheMisses: number;
    cacheWritten: number;
    chunksUpserted: number;
    prunedChunks: number;
    embedMs: number;
    dbMs: number;
  }> = [];

  const lat = new SampledLatency(5000);
  let totalEmbedMs = 0;
  let totalDbMs = 0;
  let totalCacheHits = 0;
  let totalCacheMisses = 0;
  let totalChunksUpserted = 0;
  let totalCacheWritten = 0;
  let totalPruned = 0;

  let deltaMsgAcc = 0;
  let deltaBytesAcc = 0;
  const startedAt = nowMs();

  try {
    for (let tick = 1; tick <= seconds; tick += 1) {
      const toAppend = mps;
      for (let i = 0; i < toAppend; i += 1) {
        const role: "user" | "assistant" = msgIdx % 2 === 0 ? "user" : "assistant";
        const rec: SessionJsonlMessageRecord = {
          type: "message",
          message: {
            role,
            content: randomText({ chars, seed: `msg:${msgIdx}` }),
          },
        };
        messages.push(rec);
        deltaMsgAcc += 1;
        deltaBytesAcc += Buffer.byteLength(JSON.stringify(rec) + "\n", "utf-8");
        msgIdx += 1;
      }

      const shouldSync =
        (deltaMessages > 0 && deltaMsgAcc >= deltaMessages) ||
        (deltaBytes > 0 && deltaBytesAcc >= deltaBytes) ||
        (deltaMessages === 0 && deltaBytes === 0);

      if (!shouldSync) {
        continue;
      }

      const jsonl = serializeSessionJsonl(messages);
      const session = buildSessionEntryFromJsonl(jsonl);
      const chunks = chunkMarkdownLikeOpenClaw(session.content, { tokens: chunkTokens, overlap: chunkOverlap });
      remapChunkLinesLikeOpenClaw(chunks, session.lineMap);

      const chunkTextsByHash = new Map<string, string>();
      for (const c of chunks) {
        if (!chunkTextsByHash.has(c.hash)) {
          chunkTextsByHash.set(c.hash, c.text);
        }
      }
      const uniqueHashes = Array.from(chunkTextsByHash.keys());

      // Warmup: embed a few misses once the first sync runs.
      if (warmupBatches > 0 && syncRuns.length === 0) {
        const warm = splitIntoBatches(uniqueHashes.slice(0, warmupBatches * batchSize), batchSize)
          .map((hs) => hs.map((h) => chunkTextsByHash.get(h) ?? ""));
        for (const batch of warm) {
          if (batch.length === 0) continue;
          await embedder.embedBatch(batch);
        }
      }

      const t0 = nowMs();
      const cacheRes = await embedMissingWithCache({
        embedder,
        bridge,
        dbPath,
        providerKey,
        chunkHashes: uniqueHashes,
        chunkTextsByHash,
        batchSize,
        concurrency,
      });
      const t1 = nowMs();

      const upsertRes = await bridge.chunksUpsert({
        dbPath,
        sessionId,
        roundId: tick,
        updatedAtMs: epochMs(),
        chunks: chunks.map((c) => ({
          hash: c.hash,
          startLine: c.startLine,
          endLine: c.endLine,
          text: c.text,
          embeddingHash: c.hash,
        })),
      });
      const t2 = nowMs();

      let pruned = 0;
      if (pruneMissing) {
        const pruneRes = await bridge.chunksPruneMissing({
          dbPath,
          sessionId,
          keepHashes: chunks.map((c) => c.hash),
        });
        pruned = pruneRes.deleted ?? 0;
      }
      const t3 = nowMs();

      const dt = t3 - t0;
      lat.add(dt);

      const embedMs = cacheRes.embedMs;
      const dbMs = cacheRes.dbMs + (t2 - t1) + (pruneMissing ? t3 - t2 : 0);

      totalEmbedMs += embedMs;
      totalDbMs += dbMs;
      totalCacheHits += cacheRes.hits;
      totalCacheMisses += cacheRes.misses;
      totalCacheWritten += cacheRes.writtenCache;
      totalChunksUpserted += upsertRes.written ?? 0;
      totalPruned += pruned;

      syncRuns.push({
        tick,
        appendedMessages: deltaMsgAcc,
        appendedBytes: deltaBytesAcc,
        sessionMessages: session.totalMessages,
        chunks: chunks.length,
        uniqueChunkHashes: uniqueHashes.length,
        cacheHits: cacheRes.hits,
        cacheMisses: cacheRes.misses,
        cacheWritten: cacheRes.writtenCache,
        chunksUpserted: upsertRes.written ?? 0,
        prunedChunks: pruned,
        embedMs,
        dbMs,
      });

      // Reset delta after sync.
      deltaMsgAcc = 0;
      deltaBytesAcc = 0;
    }
  } finally {
    await bridge.close();
  }

  const elapsedMs = nowMs() - startedAt;

  const meanBatchMs = syncRuns.length > 0 ? (totalEmbedMs + totalDbMs) / syncRuns.length : 0;
  const maxMs = 0;
  const totalTexts = totalCacheHits + totalCacheMisses;
  const textsPerSec = elapsedMs > 0 ? (totalTexts / elapsedMs) * 1000 : 0;
  const batchesPerSec = elapsedMs > 0 ? (syncRuns.length / elapsedMs) * 1000 : 0;

  const out = {
    kind: "highfreq-chat",
    provider: { id: embedder.id, model: embedder.model, dims: embedder.dims },
    config: {
      seconds,
      mps,
      chars,
      deltaMessages,
      deltaBytes,
      chunkTokens,
      chunkOverlap,
      batchSize,
      concurrency,
      dbPath,
      resetDb,
      pruneMissing,
    },
    totals: {
      syncRuns: syncRuns.length,
      embeddedTexts: totalTexts,
      cacheHits: totalCacheHits,
      cacheMisses: totalCacheMisses,
      cacheWritten: totalCacheWritten,
      chunksUpserted: totalChunksUpserted,
      prunedChunks: totalPruned,
      elapsedMs,
      textsPerSec,
      batchesPerSec,
      meanBatchMs,
      maxBatchMs: maxMs,
      embedMs: totalEmbedMs,
      dbMs: totalDbMs,
    },
    latency: lat.summary(),
    syncRuns,
  };
  console.log(JSON.stringify(out, null, 2));
}

main().catch((err) => {
  console.error(String(err));
  process.exitCode = 1;
});
