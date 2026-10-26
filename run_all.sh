#!/usr/bin/env bash
# One-shot script that produces every deliverable (needs GROQ_API_KEY in .env).
set -e
python main.py index ./data/Bid1 ./data/Bid2
python main.py extract --bid ./data/Bid1
python main.py extract --bid ./data/Bid2
python main.py eval
python main.py qa-log
python main.py compare Bid1 Bid2 > outputs/comparison.md
echo "Done. See outputs/ (JSON + summaries + qa_log.md + comparison.md), eval/results.md and runs/<run_id>/trace.jsonl"
