"""
Emergence Safety Experiment — Bulk Runner
Runs all 500 sentence pairs against the local server
Saves full results to results.json and summary to summary.json
Estimated time: 4-6 hours on GPU
"""

import json
import time
import requests
from datetime import datetime
from pairs import get_all_pairs

SERVER   = "http://localhost:5000"
DELAY    = 0.5   # seconds between requests — prevents server overload
SAVE_EVERY = 10  # save progress every N pairs in case of interruption

def run_pair(safe, unsafe):
    try:
        res = requests.post(
            f"{SERVER}/safety_compare",
            json={"safe": safe, "unsafe": unsafe},
            timeout=300
        )
        if res.status_code == 200:
            return res.json()
        else:
            return {"error": f"HTTP {res.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def run_experiment():
    pairs     = get_all_pairs()
    total     = len(pairs)
    results   = []
    errors    = []
    start     = datetime.now()

    print(f"Emergence Safety Experiment")
    print(f"Model: Phi-3 Mini 4k Instruct")
    print(f"Total pairs: {total}")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Estimated time: {total * 35 / 3600:.1f} hours")
    print("-" * 60)

    start_from = len(results)
    print(f"Resuming from pair {start_from + 1}")
    for i, pair in enumerate(pairs):
        if i < start_from:
            continue
        category = pair["category"]
        safe     = pair["safe"]
        unsafe   = pair["unsafe"]

        print(f"[{i+1}/{total}] {category[:20]:<20} | ", end="", flush=True)

        result = run_pair(safe, unsafe)

        if "error" in result:
            print(f"ERROR: {result['error']}")
            errors.append({"index": i, "pair": pair, "error": result["error"]})
        else:
            alarm = result.get("alarm_layer")
            div_max = max(result.get("layer_divergence", [0]))
            print(f"alarm={str(alarm):<6} max_div={div_max:.4f}")

            results.append({
                "index":           i,
                "category":        category,
                "safe":            safe,
                "unsafe":          unsafe,
                "alarm_layer":     alarm,
                "layer_divergence":result.get("layer_divergence", []),
                "layer_divergence_norm": result.get("layer_divergence_norm", []),
                "token_drivers":   result.get("token_drivers", []),
                "n_layers":        result.get("n_layers", 0),
                "model":           result.get("model", "")
            })

        # Save progress periodically
        if (i + 1) % SAVE_EVERY == 0:
            save_progress(results, errors, i + 1, total, start)

        time.sleep(DELAY)

    # Final save
    save_progress(results, errors, total, total, start)
    generate_summary(results, errors, start)

    print("-" * 60)
    print(f"Complete. {len(results)} successful, {len(errors)} errors.")
    print(f"Results saved to results.json")
    print(f"Summary saved to summary.json")


def save_progress(results, errors, done, total, start):
    data = {
        "metadata": {
            "model":       "microsoft/Phi-3-mini-4k-instruct",
            "total_pairs": total,
            "completed":   done,
            "errors":      len(errors),
            "started":     start.isoformat(),
            "saved":       datetime.now().isoformat()
        },
        "results": results,
        "errors":  errors
    }
    with open("results.json", "w") as f:
        json.dump(data, f, indent=2)


def generate_summary(results, errors, start):
    if not results:
        print("No results to summarize")
        return

    from collections import defaultdict

    # Per-category statistics
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    category_stats = {}
    for cat, cat_results in by_category.items():
        alarms       = [r["alarm_layer"] for r in cat_results if r["alarm_layer"] is not None]
        no_alarm     = [r for r in cat_results if r["alarm_layer"] is None]
        max_divs     = [max(r["layer_divergence"]) if r["layer_divergence"] else 0 for r in cat_results]
        alarm_layers = alarms

        category_stats[cat] = {
            "total":               len(cat_results),
            "alarm_detected":      len(alarms),
            "no_alarm":            len(no_alarm),
            "detection_rate":      round(len(alarms) / len(cat_results), 4),
            "mean_max_divergence": round(sum(max_divs) / len(max_divs), 6),
            "alarm_layer_values":  sorted(alarm_layers),
            "most_common_alarm_layer": max(set(alarm_layers), key=alarm_layers.count) if alarm_layers else None,
        }

    # Overall statistics
    all_alarms    = [r["alarm_layer"] for r in results if r["alarm_layer"] is not None]
    all_no_alarm  = [r for r in results if r["alarm_layer"] is None]
    all_max_divs  = [max(r["layer_divergence"]) if r["layer_divergence"] else 0 for r in results]

    # Layer distribution — how often each layer was the alarm layer
    layer_counts = defaultdict(int)
    for a in all_alarms:
        layer_counts[a] += 1

    overall = {
        "total_pairs":        len(results),
        "alarm_detected":     len(all_alarms),
        "no_alarm":           len(all_no_alarm),
        "detection_rate":     round(len(all_alarms) / len(results), 4),
        "mean_max_divergence":round(sum(all_max_divs) / len(all_max_divs), 6),
        "layer_distribution": dict(sorted(layer_counts.items())),
        "most_common_alarm_layer": max(set(all_alarms), key=all_alarms.count) if all_alarms else None,
        "errors":             len(errors),
        "duration_minutes":   round((datetime.now() - start).total_seconds() / 60, 1)
    }

    summary = {
        "metadata": {
            "model":   "microsoft/Phi-3-mini-4k-instruct",
            "date":    datetime.now().isoformat(),
            "hypothesis": "Safety discrimination is localized to a consistent alarm layer rather than distributed"
        },
        "overall":          overall,
        "by_category":      category_stats,
    }

    with open("summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print(f"Overall detection rate:     {overall['detection_rate']*100:.1f}%")
    print(f"Most common alarm layer:    {overall['most_common_alarm_layer']}")
    print(f"Layer distribution:         {dict(list(sorted(layer_counts.items(), key=lambda x: -x[1])[:5]))}")
    print()
    print(f"{'Category':<25} {'Detection%':>10} {'Alarm Layer':>12}")
    print("-" * 50)
    for cat, stats in sorted(category_stats.items(), key=lambda x: x[1]["detection_rate"]):
        print(f"{cat:<25} {stats['detection_rate']*100:>9.1f}% {str(stats['most_common_alarm_layer']):>12}")


if __name__ == "__main__":
    # Check server is running first
    try:
        res = requests.get(f"{SERVER}/health", timeout=5)
        if res.status_code == 200:
            data = res.json()
            print(f"Server OK — {data.get('model', 'unknown')}")
        else:
            print("Server returned error. Is server.py running?")
            exit(1)
    except:
        print("Cannot connect to server. Run: py -3.12 server.py")
        exit(1)

    run_experiment()
