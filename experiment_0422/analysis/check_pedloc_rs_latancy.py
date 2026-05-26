import json
import math
import time
from statistics import mean, median


JSON_PATH = "log_20260422_224621_mode2_10.json"


def calc_channel_quality(cir):
    """
    PedLoc-like CIR-based channel quality.

    Qch = 1 / (PEmax - PEavg)

    PEavg: 전체 CIR 평균 기반 energy
    PEmax: top-3 peak 평균 기반 energy
    """
    if cir is None or len(cir) == 0:
        return None

    EPS = 1e-9
    TOP_K = 3

    cir = [float(x) for x in cir]
    n = len(cir)

    avg_all = sum(cir) / n

    top_values = sorted(cir, reverse=True)[:TOP_K]
    avg_top = sum(top_values) / len(top_values)

    pe_avg = 10.0 * math.log10(avg_all + EPS)
    pe_max = 10.0 * math.log10(avg_top + EPS)

    diff = pe_max - pe_avg

    if diff <= EPS:
        diff = EPS

    qch = 1.0 / diff

    return qch


def select_root_pedloc_like(record):
    adv = record.get("adv", {})
    qch_dict = {}

    for anchor_str, adv_info in adv.items():
        anchor = int(anchor_str)
        cir = adv_info.get("cir")

        qch = calc_channel_quality(cir)
        if qch is None:
            continue

        qch_dict[anchor] = qch

    if not qch_dict:
        return None, qch_dict

    selected_root = max(qch_dict, key=qch_dict.get)
    return selected_root, qch_dict


def summarize(values):
    if not values:
        return None

    return {
        "n": len(values),
        "avg_ms": mean(values),
        "median_ms": median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def print_summary(name, values):
    s = summarize(values)

    print(f"\n[{name}]")
    if s is None:
        print("  no valid samples")
        return

    print(f"  n      = {s['n']}")
    print(f"  avg    = {s['avg_ms']:.6f} ms")
    print(f"  median = {s['median_ms']:.6f} ms")
    print(f"  min    = {s['min_ms']:.6f} ms")
    print(f"  max    = {s['max_ms']:.6f} ms")


with open(JSON_PATH, "r", encoding="utf-8") as f:
    records = json.load(f)


latencies_ms = []
match_count = 0
valid_count = 0
mismatch_examples = []

for record in records:
    if "adv" not in record or not record["adv"]:
        continue

    original_root = record.get("root_anchor")

    start = time.perf_counter()
    selected_root, qch_dict = select_root_pedloc_like(record)
    latency_ms = (time.perf_counter() - start) * 1000.0

    if selected_root is None:
        continue

    latencies_ms.append(latency_ms)
    valid_count += 1

    if original_root == selected_root:
        match_count += 1
    else:
        if len(mismatch_examples) < 20:
            mismatch_examples.append({
                "seq": record.get("seq"),
                "original_root": original_root,
                "selected_root": selected_root,
                "qch": qch_dict,
            })


print(f"Total records       : {len(records)}")
print(f"Valid ADV records   : {valid_count}")
print(f"Root match count    : {match_count}/{valid_count}")

if valid_count > 0:
    print(f"Root match ratio    : {match_count / valid_count * 100:.2f}%")

print_summary("PedLoc-like root selection latency", latencies_ms)

if mismatch_examples:
    print("\n[Mismatch examples]")
    for ex in mismatch_examples:
        qch_round = {k: round(v, 6) for k, v in sorted(ex["qch"].items())}
        print(
            f"  seq={ex['seq']}, "
            f"original={ex['original_root']}, "
            f"selected={ex['selected_root']}, "
            f"qch={qch_round}"
        )