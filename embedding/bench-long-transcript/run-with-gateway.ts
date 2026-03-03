#!/usr/bin/env -S npx --yes tsx@4.21.0

import { spawn, type ChildProcess } from "node:child_process";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import path from "node:path";
import os from "node:os";
import fs from "node:fs";

function usage(): string {
  return `Run bench-long-transcript with a real OpenClaw Gateway process

This orchestrates:
  1) Start OpenClaw Gateway (ws://127.0.0.1:18789 by default)
  2) Spawn a best-effort headless TUI connection process (non-interactive)
  3) Run the embedding benchmark (bench.ts)
  4) Cleanup all spawned processes

Usage:
  npx --yes tsx@4.21.0 openclaw-playground/embedding/bench-long-transcript/run-with-gateway.ts [runner options] -- [bench options]

Runner options:
  --url=<wsUrl>         Gateway WebSocket URL (default: ws://127.0.0.1:18789)
  --port=<n>            Gateway port (default: 18789)
  --stateDir=<path>     OPENCLAW_STATE_DIR for the spawned gateway (default: temp dir)
  --spawnTui=<0|1>      Try to spawn a headless 'openclaw tui' client (default: 1)
  --tuiSession=<key>    TUI session key (default: main)
  --gatewayDev=<0|1>    Use 'gateway run --dev' (default: 1)
  --keepAliveMs=<ms>    Extra time to keep gateway+tui alive after bench (default: 0)

Bench options:
  Everything after "--" is passed to bench.ts.

Examples:
  npx --yes tsx@4.21.0 openclaw-playground/embedding/bench-long-transcript/run-with-gateway.ts -- --provider=fake --mb=32

Notes:
  - openclaw tui supports --url and --session. It does not accept --agent.
  - The embedding benchmark does not depend on the gateway; this is to simulate
    a realistic "gateway is running + a client is connected" environment.
`;
}

type Parsed = {
  flags: Record<string, string | boolean>;
  benchArgv: string[];
};

function parseRunnerArgs(argv: string[]): Parsed {
  const dashDash = argv.indexOf("--");
  const runnerArgv = dashDash >= 0 ? argv.slice(0, dashDash) : argv;
  const benchArgv = dashDash >= 0 ? argv.slice(dashDash + 1) : [];

  const flags: Record<string, string | boolean> = {};
  for (const raw of runnerArgv) {
    if (!raw.startsWith("--")) continue;
    const eq = raw.indexOf("=");
    if (eq === -1) {
      flags[raw.slice(2)] = true;
    } else {
      flags[raw.slice(2, eq)] = raw.slice(eq + 1);
    }
  }
  return { flags, benchArgv };
}

function readString(flags: Record<string, string | boolean>, key: string, def?: string): string | undefined {
  const v = flags[key];
  if (typeof v === "string") return v;
  if (v === true) return "true";
  return def;
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
  retries: number;
  delayMs: number;
}): Promise<void> {
  for (let i = 1; i <= params.retries; i += 1) {
    const child = spawn(process.execPath, ["./openclaw.mjs", "gateway", "health"], {
      cwd: params.openclawRepoDir,
      env: params.env,
      stdio: ["ignore", "ignore", "ignore"],
    });

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

async function terminate(proc: ChildProcess | null | undefined, name: string): Promise<void> {
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

  process.stderr.write(`[runner] forced kill: ${name} pid=${pid}\n`);
}

async function main(): Promise<void> {
  const parsed = parseRunnerArgs(process.argv.slice(2));
  if (parsed.flags.help) {
    console.log(usage());
    return;
  }

  const here = path.dirname(fileURLToPath(import.meta.url));
  const openclawRepoDir = path.resolve(here, "../../../openclaw-epeshared");

  const url = readString(parsed.flags, "url", "ws://127.0.0.1:18789")!;
  const port = readInt(parsed.flags, "port", 18789);
  const spawnTui = readBool(parsed.flags, "spawnTui", true);
  const tuiSession = readString(parsed.flags, "tuiSession", "main")!;
  const gatewayDev = readBool(parsed.flags, "gatewayDev", true);
  const keepAliveMs = readInt(parsed.flags, "keepAliveMs", 0);

  const stateDir =
    readString(parsed.flags, "stateDir") ??
    path.join(os.tmpdir(), `openclaw-bench-long-transcript-${process.pid}`);

  // Important: force an isolated config path so OpenClaw doesn't auto-select an
  // existing config file from default locations (e.g. ~/.openclaw/openclaw.json)
  // which may carry a different gateway.auth.token and override our env token.
  const configPath = path.join(stateDir, "openclaw.json");

  const gatewayToken =
    (process.env.OPENCLAW_GATEWAY_TOKEN && process.env.OPENCLAW_GATEWAY_TOKEN.trim())
      ? process.env.OPENCLAW_GATEWAY_TOKEN.trim()
      : crypto.randomUUID();

  const env: NodeJS.ProcessEnv = {
    ...process.env,
    OPENCLAW_STATE_DIR: stateDir,
    OPENCLAW_CONFIG_PATH: configPath,
    // Make the runner deterministic and avoid interactive wizards.
    OPENCLAW_GATEWAY_TOKEN: gatewayToken,
  };

  let gatewayProc: ChildProcess | undefined;
  let tuiProc: ChildProcess | undefined;

  const benchScript = path.join(here, "bench.ts");

  const cleanup = async () => {
    await terminate(tuiProc, "tui");
    await terminate(gatewayProc, "gateway");
  };

  process.on("SIGINT", () => {
    void cleanup().finally(() => process.exit(130));
  });
  process.on("SIGTERM", () => {
    void cleanup().finally(() => process.exit(143));
  });

  try {
    await fs.promises.mkdir(stateDir, { recursive: true });

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
      gatewayToken,
    ];
    if (gatewayDev) {
      gatewayArgs.splice(3, 0, "--dev");
    }

    gatewayProc = spawnLogged("gateway", process.execPath, gatewayArgs, {
      cwd: openclawRepoDir,
      env,
      stdio: ["ignore", "pipe", "pipe"],
    });

    await waitForGatewayHealth({ openclawRepoDir, env, retries: 60, delayMs: 500 });
    process.stderr.write(`[runner] gateway healthy: ${url} (stateDir=${stateDir})\n`);

    if (spawnTui) {
      // Best-effort: TUI usually expects a real TTY. We spawn it headlessly to
      // simulate “a terminal client is connected” without taking over stdout.
      tuiProc = spawnLogged(
        "tui",
        process.execPath,
        [
          "./openclaw.mjs",
          "tui",
          "--url",
          url,
          "--token",
          gatewayToken,
          "--session",
          tuiSession,
          "--history-limit",
          "1",
        ],
        {
          cwd: openclawRepoDir,
          env,
          stdio: ["ignore", "ignore", "pipe"],
        },
      );
      await sleep(500);
    }

    // Run the benchmark.
    const benchArgs = ["--yes", "tsx@4.21.0", benchScript, ...parsed.benchArgv];
    const bench = spawn("npx", benchArgs, {
      cwd: here,
      env: process.env,
      stdio: "inherit",
    });

    const benchExitCode: number = await new Promise((resolve) => {
      bench.on("exit", (code) => resolve(code ?? 1));
      bench.on("error", () => resolve(1));
    });

    if (keepAliveMs > 0) {
      await sleep(keepAliveMs);
    }

    await cleanup();
    process.exitCode = benchExitCode;
  } catch (err) {
    process.stderr.write(`[runner] error: ${err instanceof Error ? err.message : String(err)}\n`);
    await cleanup();
    process.exitCode = 1;
  }
}

main().catch((err) => {
  process.stderr.write(
    `[runner] fatal: ${err instanceof Error ? err.stack ?? err.message : String(err)}\n`,
  );
  process.exitCode = 1;
});
