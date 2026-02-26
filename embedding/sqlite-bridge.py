#!/usr/bin/env python
"""A tiny stdin/stdout JSON bridge for SQLite using Python stdlib.

This avoids needing node:sqlite (not in Node 20) or sqlite3 CLI.

Protocol: newline-delimited JSON.
Request:  {"id": 1, "method": "init", "params": {...}}
Response: {"id": 1, "ok": true, "result": {...}}
Error:    {"id": 1, "ok": false, "error": "..."}
"""

from __future__ import annotations

import json
import sqlite3
import sys
from typing import Any, Dict, List, Optional


class Bridge:
    def __init__(self) -> None:
        self._db_path: Optional[str] = None
        self._conn: Optional[sqlite3.Connection] = None

    def _ensure_conn(self, db_path: str) -> sqlite3.Connection:
        if self._conn is not None and self._db_path == db_path:
            return self._conn
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._conn.row_factory = sqlite3.Row
        return self._conn

    def init(self, params: Dict[str, Any]) -> Dict[str, Any]:
        db_path = str(params["dbPath"])
        reset = bool(params.get("reset", False))
        conn = self._ensure_conn(db_path)
        if reset:
            conn.executescript(
                """
                DROP TABLE IF EXISTS embedding_cache;
                DROP TABLE IF EXISTS chunks;
                """
            )
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
              provider TEXT NOT NULL,
              model TEXT NOT NULL,
              provider_key TEXT NOT NULL,
              hash TEXT NOT NULL,
              embedding_json TEXT NOT NULL,
              updated_at_ms INTEGER NOT NULL,
              PRIMARY KEY (provider, model, provider_key, hash)
            );

            CREATE TABLE IF NOT EXISTS chunks (
              session_id TEXT NOT NULL,
              chunk_hash TEXT NOT NULL,
              start_line INTEGER NOT NULL,
              end_line INTEGER NOT NULL,
              text TEXT NOT NULL,
              embedding_hash TEXT NOT NULL,
              round_id INTEGER NOT NULL,
              updated_at_ms INTEGER NOT NULL,
              PRIMARY KEY (session_id, chunk_hash)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_session_round ON chunks(session_id, round_id);
            CREATE INDEX IF NOT EXISTS idx_cache_hash ON embedding_cache(hash);
            """
        )
        conn.commit()
        return {"dbPath": db_path}

    def cache_check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        conn = self._ensure_conn(str(params["dbPath"]))
        provider = str(params["provider"])
        model = str(params["model"])
        provider_key = str(params["providerKey"])
        hashes = params.get("hashes") or []
        if not isinstance(hashes, list):
            raise ValueError("hashes must be a list")
        if len(hashes) == 0:
            return {"hits": []}
        # Chunk into <=999 to satisfy SQLite parameter limit.
        hits: List[str] = []
        for i in range(0, len(hashes), 900):
            batch = hashes[i : i + 900]
            q_marks = ",".join(["?"] * len(batch))
            rows = conn.execute(
                f"SELECT hash FROM embedding_cache WHERE provider=? AND model=? AND provider_key=? AND hash IN ({q_marks})",
                [provider, model, provider_key, *batch],
            ).fetchall()
            hits.extend([str(r["hash"]) for r in rows])
        return {"hits": hits}

    def cache_put(self, params: Dict[str, Any]) -> Dict[str, Any]:
        conn = self._ensure_conn(str(params["dbPath"]))
        provider = str(params["provider"])
        model = str(params["model"])
        provider_key = str(params["providerKey"])
        updated_at_ms = int(params.get("updatedAtMs", 0))
        items = params.get("items") or []
        if not isinstance(items, list):
            raise ValueError("items must be a list")
        cur = conn.cursor()
        written = 0
        cur.execute("BEGIN")
        for item in items:
            h = str(item["hash"])
            emb = str(item["embeddingJson"])
            cur.execute(
                """
                INSERT INTO embedding_cache(provider, model, provider_key, hash, embedding_json, updated_at_ms)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(provider, model, provider_key, hash)
                DO UPDATE SET embedding_json=excluded.embedding_json, updated_at_ms=excluded.updated_at_ms
                """,
                (provider, model, provider_key, h, emb, updated_at_ms),
            )
            written += 1
        conn.commit()
        return {"written": written}

    def chunks_upsert(self, params: Dict[str, Any]) -> Dict[str, Any]:
        conn = self._ensure_conn(str(params["dbPath"]))
        session_id = str(params["sessionId"])
        round_id = int(params["roundId"])
        updated_at_ms = int(params.get("updatedAtMs", 0))
        chunks = params.get("chunks") or []
        if not isinstance(chunks, list):
            raise ValueError("chunks must be a list")

        cur = conn.cursor()
        cur.execute("BEGIN")
        written = 0
        for c in chunks:
            cur.execute(
                """
                INSERT INTO chunks(session_id, chunk_hash, start_line, end_line, text, embedding_hash, round_id, updated_at_ms)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id, chunk_hash)
                DO UPDATE SET
                  start_line=excluded.start_line,
                  end_line=excluded.end_line,
                  text=excluded.text,
                  embedding_hash=excluded.embedding_hash,
                  round_id=excluded.round_id,
                  updated_at_ms=excluded.updated_at_ms
                """,
                (
                    session_id,
                    str(c["hash"]),
                    int(c["startLine"]),
                    int(c["endLine"]),
                    str(c["text"]),
                    str(c["embeddingHash"]),
                    round_id,
                    updated_at_ms,
                ),
            )
            written += 1
        conn.commit()
        return {"written": written}

    def chunks_prune_missing(self, params: Dict[str, Any]) -> Dict[str, Any]:
        conn = self._ensure_conn(str(params["dbPath"]))
        session_id = str(params["sessionId"])
        keep_hashes = params.get("keepHashes") or []
        if not isinstance(keep_hashes, list):
            raise ValueError("keepHashes must be a list")
        if len(keep_hashes) == 0:
            cur = conn.execute("DELETE FROM chunks WHERE session_id=?", (session_id,))
            conn.commit()
            return {"deleted": int(cur.rowcount)}
        deleted = 0
        # Delete in chunks to avoid huge IN lists.
        # Approach: mark all as candidates, then keep by IN list per batch.
        # For simplicity: delete rows not in keep set using a temp table.
        cur = conn.cursor()
        cur.execute("BEGIN")
        cur.execute("DROP TABLE IF EXISTS _keep_hashes")
        cur.execute("CREATE TEMP TABLE _keep_hashes(hash TEXT PRIMARY KEY)")
        for i in range(0, len(keep_hashes), 900):
            batch = keep_hashes[i : i + 900]
            cur.executemany("INSERT OR IGNORE INTO _keep_hashes(hash) VALUES(?)", [(str(h),) for h in batch])
        cur.execute(
            "DELETE FROM chunks WHERE session_id=? AND chunk_hash NOT IN (SELECT hash FROM _keep_hashes)",
            (session_id,),
        )
        deleted = int(cur.rowcount)
        conn.commit()
        return {"deleted": deleted}

    def close(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None
        self._db_path = None
        return {"closed": True}


def main() -> None:
    bridge = Bridge()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError("params must be an object")

            if method == "init":
                result = bridge.init(params)
            elif method == "cacheCheck":
                result = bridge.cache_check(params)
            elif method == "cachePut":
                result = bridge.cache_put(params)
            elif method == "chunksUpsert":
                result = bridge.chunks_upsert(params)
            elif method == "chunksPruneMissing":
                result = bridge.chunks_prune_missing(params)
            elif method == "close":
                result = bridge.close(params)
            else:
                raise ValueError(f"unknown method: {method}")

            resp = {"id": req_id, "ok": True, "result": result}
        except Exception as e:
            resp = {"id": req.get("id") if isinstance(req, dict) else None, "ok": False, "error": str(e)}

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
