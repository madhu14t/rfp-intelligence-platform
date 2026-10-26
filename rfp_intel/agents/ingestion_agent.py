"""Ingestion Agent: detects file types, parses, cleans, attaches metadata and triggers indexing."""
from __future__ import annotations

from pathlib import Path

from rfp_intel.agents.base import Agent


class IngestionAgent(Agent):
    name = "ingestion_agent"

    def run(self, folder: str | Path, bid_id: str | None = None, force: bool = False) -> dict:
        folder = Path(folder)
        bid_id = bid_id or folder.name
        with self.tracer.step(self.name, "index_folder", input={"folder": str(folder), "bid_id": bid_id, "force": force}) as ctx:
            if not force and bid_id in self.engine.list_bids():
                ctx["output"] = {"bid_id": bid_id, "status": "already_indexed"}
                return ctx["output"]
            result = self.engine.index_folder(folder, bid_id)
            ctx["output"] = result
            return result
