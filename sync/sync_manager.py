r"""
sync/sync_manager.py — Phase 4 sync mechanism

Background, opportunistic re-ingestion of the knowledge base when the
device has internet access. This is the "offline-first with opportunistic
sync" piece from the project abstract: the system works fully offline off
the pre-loaded knowledge base, and whenever connectivity happens to be
available it checks a configured list of authoritative source URLs,
downloads anything whose content changed (compared by SHA-256 checksum),
and re-ingests just that one document — no full re-ingest, no interruption
to normal offline queries.

HOW IT WORKS
    1. sync/sources.json maps a source_name (must match the filename stem
       used by phase_one/ingest.py, e.g. "cdc_40918_DS1") to a direct
       download URL for the authoritative version of that document.
    2. Every poll interval (default 30 min), if online:
        - download each configured URL
        - compute its SHA-256 checksum
        - compare to the checksum recorded in sync/manifest.json from the
          last successful sync
        - if different (or first time seen): write the new file into
          knowledge_base/, delete the old chunks for that source from
          LanceDB, chunk + embed + add the new version, update the manifest
    3. If offline, skip silently and try again next interval.

CONFIGURE YOUR SOURCES
    Edit sync/sources.json. It ships as a template with no real URLs filled
    in — a wrong URL would silently pull in the wrong document, so you need
    to supply and verify the real download links for whichever WHO / NDMA /
    Red Cross / CDC documents you want auto-synced.

HOW TO RUN
    One-off check, good for testing without waiting 30 minutes:
        python sync/sync_manager.py --once

    Continuous background polling (Ctrl+C to stop):
        python sync/sync_manager.py --daemon

    Custom interval (seconds) and paths:
        python sync/sync_manager.py --daemon --interval 900 --db data/lancedb

    From code, e.g. started alongside phase_three/chat.py:
        from sync.sync_manager import SyncManager
        sync_mgr = SyncManager(db_path="data/lancedb")
        sync_mgr.start()   # non-blocking, runs in a background thread
        ...
        sync_mgr.stop()
"""

import sys
import json
import socket
import hashlib
import argparse
import threading
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase_one.ingest import extract_text_from_files
from phase_one.chunker import chunk_document
from phase_one.embedder import Embedder
from phase_one.vector_store import VectorStore

DEFAULT_SOURCES_PATH  = Path(__file__).parent / "sources.json"
DEFAULT_MANIFEST_PATH = Path(__file__).parent / "manifest.json"
DEFAULT_KB_DIR         = Path("knowledge_base")
DEFAULT_DB_PATH        = "data/lancedb"
DEFAULT_POLL_INTERVAL  = 1800  # 30 minutes, per the README spec
DOWNLOAD_TIMEOUT_SEC   = 15
CONNECTIVITY_CHECK_HOST = ("8.8.8.8", 53)  # Google DNS — checked via raw
                                            # socket connect so this doesn't
                                            # depend on any one website's
                                            # uptime, just basic internet.
CONNECTIVITY_TIMEOUT_SEC = 3.0


def is_online(timeout: float = CONNECTIVITY_TIMEOUT_SEC) -> bool:
    """Cheap connectivity check: can we open a socket to a DNS resolver."""
    try:
        with socket.create_connection(CONNECTIVITY_CHECK_HOST, timeout=timeout):
            return True
    except OSError:
        return False


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str, timeout: int = DOWNLOAD_TIMEOUT_SEC) -> bytes | None:
    try:
        req = urllib.request.Request(
            url,
            headers={
                # Some servers (CDC's included) reject requests with an
                # obviously non-browser User-Agent. This is just an honest
                # browser-like string, not spoofing anything malicious.
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "application/pdf,text/html,*/*",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout, ValueError) as e:
        print(f"[Sync] Download failed for {url}: {e}")
        return None


class SyncManager:
    def __init__(
        self,
        sources_path: str | Path = DEFAULT_SOURCES_PATH,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        knowledge_base_dir: str | Path = DEFAULT_KB_DIR,
        db_path: str = DEFAULT_DB_PATH,
        poll_interval_sec: int = DEFAULT_POLL_INTERVAL,
    ):
        self.sources_path       = Path(sources_path)
        self.manifest_path      = Path(manifest_path)
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.db_path            = db_path
        self.poll_interval_sec  = poll_interval_sec

        self._thread      = None
        self._stop_event  = threading.Event()
        self._embedder    = None  # lazy — only load the model if a sync actually runs
        self._store       = None

    # ---------- config / state I/O ----------

    def _load_sources(self) -> dict:
        if not self.sources_path.exists():
            print(f"[Sync] No sources file at {self.sources_path} — nothing to sync.")
            return {}
        raw = json.loads(self.sources_path.read_text(encoding="utf-8"))
        raw.pop("_readme", None)
        return raw

    def _load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _save_manifest(self, manifest: dict):
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ---------- the actual sync ----------

    def check_and_sync_once(self) -> dict:
        """
        Runs a single sync pass. Safe to call directly (e.g. for a manual
        "sync now" button) or from the background loop. Returns a summary
        dict instead of raising, so a bad run doesn't kill the poll loop.
        """
        if not is_online():
            print("[Sync] Offline — skipping this check.")
            return {"status": "offline", "updated": [], "unchanged": [], "failed": []}

        sources = self._load_sources()
        configured = {k: v for k, v in sources.items() if v.get("url")}
        if not configured:
            print("[Sync] No sources configured with a URL yet — see sync/sources.json.")
            return {"status": "no_sources", "updated": [], "unchanged": [], "failed": []}

        manifest = self._load_manifest()
        updated, unchanged, failed = [], [], []

        for source_name, cfg in configured.items():
            url = cfg["url"]
            filename = cfg.get("filename", f"{source_name}.pdf")

            data = download(url)
            if data is None:
                failed.append(source_name)
                continue

            checksum = sha256_of_bytes(data)
            previous = manifest.get(source_name, {}).get("checksum")

            if checksum == previous:
                unchanged.append(source_name)
                continue

            print(f"[Sync] '{source_name}' changed (or new) — re-ingesting.")
            self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
            dest_path = self.knowledge_base_dir / filename
            dest_path.write_bytes(data)

            try:
                self._reingest_source(source_name, dest_path)
            except Exception as e:
                print(f"[Sync] Re-ingest failed for '{source_name}': {e}")
                failed.append(source_name)
                continue

            manifest[source_name] = {
                "checksum": checksum,
                "url": url,
                "last_synced": datetime.now(timezone.utc).isoformat(),
            }
            updated.append(source_name)

        self._save_manifest(manifest)

        print(
            f"[Sync] Done. updated={len(updated)} unchanged={len(unchanged)} "
            f"failed={len(failed)}"
        )
        return {"status": "ok", "updated": updated, "unchanged": unchanged, "failed": failed}

    def _reingest_source(self, source_name: str, file_path: Path):
        """Re-embeds one document and swaps its chunks in LanceDB."""
        if self._embedder is None:
            self._embedder = Embedder()
        if self._store is None:
            self._store = VectorStore(self.db_path)
            self._store.init()

        text = extract_text_from_files(file_path)
        if not text:
            print(f"[Sync] '{source_name}': no extractable text, skipping re-ingest.")
            return

        chunks = chunk_document(text, source_name)
        if not chunks:
            print(f"[Sync] '{source_name}': produced no chunks, skipping re-ingest.")
            return

        vectors = self._embedder.embed_batch([c["text"] for c in chunks])
        for i, chunk in enumerate(chunks):
            chunk["vector"] = vectors[i]

        # Remove the old version's chunks first (no-op if this is a new source)
        self._store.delete_source(source_name)
        self._store.add_chunks(chunks)
        print(f"[Sync] '{source_name}': re-ingested {len(chunks)} chunks.")

    # ---------- background thread control ----------

    def start(self):
        """Starts polling in a background daemon thread. Non-blocking."""
        if self._thread and self._thread.is_alive():
            print("[Sync] Already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[Sync] Background sync started (every {self.poll_interval_sec}s).")

    def stop(self):
        """Signals the background thread to stop and waits for it to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        print("[Sync] Background sync stopped.")

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                self.check_and_sync_once()
            except Exception as e:
                print(f"[Sync] Unexpected error during sync: {e}")
            # wait() returns early if stop() is called, unlike time.sleep()
            self._stop_event.wait(self.poll_interval_sec)


def main():
    parser = argparse.ArgumentParser(description="Phase 4 — knowledge base sync manager")
    parser.add_argument("--once", action="store_true", help="Run a single sync check and exit")
    parser.add_argument("--daemon", action="store_true", help="Run continuous background polling")
    parser.add_argument("--interval", type=int, default=DEFAULT_POLL_INTERVAL,
                         help="Poll interval in seconds (default: 1800 = 30 min)")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to LanceDB folder")
    parser.add_argument("--sources", default=str(DEFAULT_SOURCES_PATH), help="Path to sources.json")
    parser.add_argument("--kb-dir", default=str(DEFAULT_KB_DIR), help="Path to knowledge_base/")
    args = parser.parse_args()

    if not args.once and not args.daemon:
        parser.print_help()
        sys.exit(1)

    manager = SyncManager(
        sources_path=args.sources,
        knowledge_base_dir=args.kb_dir,
        db_path=args.db,
        poll_interval_sec=args.interval,
    )

    if args.once:
        manager.check_and_sync_once()
        return

    manager.start()
    try:
        while True:
            threading.Event().wait(1)
    except KeyboardInterrupt:
        print("\n[Sync] Ctrl+C received, shutting down...")
        manager.stop()


if __name__ == "__main__":
    main()