import os
import json
from datetime import datetime
import numpy as np
from huggingface_hub import HfApi
import config
import data_manager as dm
from susy_qm import susy_qm_score

def normalize_scores(score_dict):
    """Normalize to [0,1] but fallback to 0 if constant."""
    scores = np.array(list(score_dict.values()))
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-12:
        return {k: 0.5 for k in score_dict}  # fallback to 0.5
    norm = (scores - min_s) / (max_s - min_s)
    return {ticker: float(norm[i]) for i, ticker in enumerate(score_dict.keys())}

def run_for_window(returns, window_days):
    if len(returns) < window_days:
        return None
    ret_window = returns.iloc[-window_days:]
    raw_scores = {}
    for ticker in ret_window.columns:
        s = susy_qm_score(ret_window[ticker])
        raw_scores[ticker] = float(s)
    # Print first few raw scores for debugging
    if raw_scores:
        first_ticker = list(raw_scores.keys())[0]
        print(f"    Sample raw score for {first_ticker}: {raw_scores[first_ticker]}")
    norm_scores = normalize_scores(raw_scores)
    sorted_norm = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
    top_etfs = [{"ticker": t, "susy_score_norm": s, "raw_score": raw_scores[t]} for t, s in sorted_norm[:config.TOP_N]]
    return {
        "window": window_days,
        "top_etfs": top_etfs,
        "all_scores_raw": raw_scores,
        "all_scores_norm": norm_scores
    }

def main():
    print("Loading master data...")
    dm.load_master_data()
    results = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "windows": config.WINDOWS,
        "universes": {}
    }
    for uni_name in config.UNIVERSES.keys():
        print(f"Processing {uni_name}...")
        returns = dm.get_universe_returns(uni_name)
        if returns.empty:
            print("  No data -> skipping")
            continue
        all_window_results = []
        best_score = -np.inf
        best_window = None
        best_data = None
        for w in config.WINDOWS:
            print(f"  Window {w} days")
            out = run_for_window(returns, w)
            if out:
                all_window_results.append(out)
                # best window = largest raw score (highest vol)
                max_raw = max(out["all_scores_raw"].values())
                if max_raw > best_score:
                    best_score = max_raw
                    best_window = w
                    best_data = out
            else:
                print(f"    Failed for window {w}")
        results["universes"][uni_name] = {
            "best_window": best_window,
            "best_window_data": best_data,
            "all_windows": all_window_results
        }
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"output/susy_qm_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")
    api = HfApi(token=config.HF_TOKEN)
    try:
        api.upload_file(
            path_or_fileobj=out_file,
            path_in_repo=os.path.basename(out_file),
            repo_id=config.OUTPUT_REPO,
            repo_type="dataset"
        )
        print(f"Uploaded to {config.OUTPUT_REPO}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
