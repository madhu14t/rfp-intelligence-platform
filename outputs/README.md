# outputs/

Generated artifacts land here. They need a Groq key, so run once after setup:

    ./run_all.sh            # or step by step:
    python main.py --bid ./data/Bid1      -> outputs/Bid1.json + Bid1_summary.md + runs/<run_id>/trace.jsonl
    python main.py --bid ./data/Bid2      -> outputs/Bid2.json + Bid2_summary.md
    python main.py qa-log                 -> outputs/qa_log.md (13 sample questions with citations)
    python main.py compare Bid1 Bid2      -> outputs/comparison.md
    python main.py eval                   -> eval/results.md (Recall@k / MRR / nDCG, 4 configurations)
