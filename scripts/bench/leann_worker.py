"""Persistent LEANN 0.3.8 search API; CLI builds the index using its own chunking."""
import json
import os
import sys
import time

# Native libraries may print diagnostics to fd 1. Keep the protocol separate.
protocol = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
from leann import LeannSearcher  # noqa: E402

searcher = LeannSearcher(sys.argv[1], use_daemon=False)
try:
    for line in sys.stdin:
        request = json.loads(line)
        start = time.perf_counter()
        if request.get("op") == "status":
            manager = searcher.passage_manager
            paths = {manager.get_passage(key)["metadata"]["file_path"]
                     for offsets in manager.offset_maps.values() for key in offsets}
            protocol.write(json.dumps({"id": request["id"], "paths": sorted(paths),
                                       "passages": len(manager)}) + "\n")
            continue
        hits = searcher.search(request["q"], top_k=200)
        protocol.write(json.dumps({"id": request["id"], "hits": [
            {"path": h.metadata["file_path"], "score": float(h.score)} for h in hits
        ], "elapsed_ms": (time.perf_counter() - start) * 1000}) + "\n")
finally:
    searcher.cleanup()
