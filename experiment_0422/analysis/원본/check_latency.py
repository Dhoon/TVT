import re
from statistics import mean, median

LOG_PATH = "log_20260422_234741_mode2_nlos.txt"

DWT_TIME_UNIT = 1 / (499.2e6 * 128.0)
DWT_TIMESTAMP_MOD = 1 << 32

def parse_time_to_us(t: str) -> int:
    hh, mm, ss, ms, us = map(int, t.split(":"))
    return (((hh * 60 + mm) * 60 + ss) * 1_000_000) + ms * 1000 + us


def diff_ms(start_us: int, end_us: int) -> float:
    day_us = 24 * 60 * 60 * 1_000_000
    if end_us < start_us:
        end_us += day_us
    return (end_us - start_us) / 1000.0


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
    if s is None:
        print(f"\n[{name}]")
        print("  no valid samples")
        return

    print(f"\n[{name}]")
    print(f"  n      = {s['n']}")
    print(f"  avg    = {s['avg_ms']:.3f} ms")
    print(f"  median = {s['median_ms']:.3f} ms")
    print(f"  min    = {s['min_ms']:.3f} ms")
    print(f"  max    = {s['max_ms']:.3f} ms")


time_pattern = r"(\d{2}:\d{2}:\d{2}:\d{3}:\d{3})"

adv_re = re.compile(time_pattern + r".*\[ADV\].*Tag\s+(\d+),\s+Seq=(\d+)")
root_re = re.compile(time_pattern + r".*\[ROOT SELECTED\]\s+Anchor\s+(\d+)\s+for\s+Tag\s+(\d+)\s+Seq\s+(\d+)")
tx_re = re.compile(time_pattern + r".*\[TX\]\s+Sent root ID\s+(\d+)")
leaf_re = re.compile(time_pattern + r".*\[Leaf\].*Anchor\s+(\d+),\s+Tag\s+(\d+),\s+Seq=(\d+)")
dstwr_re = re.compile(time_pattern + r".*\[DS-TWR\].*Anchor\s+(\d+),\s+Tag\s+(\d+),\s+Seq=(\d+).*message\s*=\s*\[(\d+),\s*(\d+)")
tof_re = re.compile(time_pattern + r".*\[TOF\]")
pos_re = re.compile(time_pattern + r".*\[POSITION\]")


epochs = []

current_epoch = None
last_dstwr_epoch = None


def new_epoch(tag, seq, adv_time):
    return {
        "tag": int(tag),
        "seq": int(seq),
        "adv_first": adv_time,
        "adv_last": adv_time,
        "root_selected": None,
        "tx_first": None,
        "tx_last": None,
        "leaf_first": None,
        "leaf_last": None,
        "dstwr": None,
        "tof": None,
        "position": None,
        "root_anchor": None,
        "dstwr_msg_diff_s": None,
    }


with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        # ADV
        m = adv_re.match(line)
        if m:
            t, tag, seq = m.groups()
            ts = parse_time_to_us(t)
            tag = int(tag)
            seq = int(seq)

            # 현재 epoch가 없거나, 이미 ROOT SELECTED 이후인데 ADV가 다시 나오면 새 epoch
            if (
                current_epoch is None
                or current_epoch["seq"] != seq
                or current_epoch["tag"] != tag
                or current_epoch["root_selected"] is not None
            ):
                current_epoch = new_epoch(tag, seq, ts)
                epochs.append(current_epoch)
            else:
                # 같은 Seq의 여러 ADV 수신
                current_epoch["adv_last"] = ts

            continue

        # ROOT SELECTED
        m = root_re.match(line)
        if m:
            t, root, tag, seq = m.groups()
            ts = parse_time_to_us(t)
            tag = int(tag)
            seq = int(seq)

            # 현재 epoch와 일치할 때만 붙임
            if (
                current_epoch is not None
                and current_epoch["tag"] == tag
                and current_epoch["seq"] == seq
            ):
                current_epoch["root_selected"] = ts
                current_epoch["root_anchor"] = int(root)

            continue

        # TX
        m = tx_re.match(line)
        if m:
            t, root = m.groups()
            ts = parse_time_to_us(t)

            # TX에는 Seq가 없으므로 직전 ROOT SELECTED epoch에 붙임
            if current_epoch is not None and current_epoch["root_selected"] is not None:
                if current_epoch["tx_first"] is None:
                    current_epoch["tx_first"] = ts
                current_epoch["tx_last"] = ts

            continue

        # Leaf
        m = leaf_re.match(line)
        if m:
            t, anchor, tag, seq = m.groups()
            ts = parse_time_to_us(t)
            tag = int(tag)
            seq = int(seq)

            if (
                current_epoch is not None
                and current_epoch["tag"] == tag
                and current_epoch["seq"] == seq
            ):
                if current_epoch["leaf_first"] is None:
                    current_epoch["leaf_first"] = ts
                current_epoch["leaf_last"] = ts

            continue

        # DS-TWR
        m = dstwr_re.match(line)
        if m:
            t, anchor, tag, seq, msg0, msg1 = m.groups()
            ts = parse_time_to_us(t)
            tag = int(tag)
            seq = int(seq)

            if (
                current_epoch is not None
                and current_epoch["tag"] == tag
                and current_epoch["seq"] == seq
            ):
                current_epoch["dstwr"] = ts
                current_epoch["dstwr_msg_diff_s"] = (int(msg1) + int(msg0)) * DWT_TIME_UNIT
                last_dstwr_epoch = current_epoch

            continue

        # TOF
        m = tof_re.match(line)
        if m:
            t = m.group(1)
            ts = parse_time_to_us(t)

            if last_dstwr_epoch is not None:
                last_dstwr_epoch["tof"] = ts

            continue

        # POSITION
        m = pos_re.match(line)
        if m:
            t = m.group(1)
            ts = parse_time_to_us(t)

            if last_dstwr_epoch is not None:
                last_dstwr_epoch["position"] = ts

            continue


root_selection_times = []
dstwr_comm_times_first_tx = []
dstwr_comm_times_last_tx = []
positioning_times = []
tof_times = []
position_after_tof_times = []
total_times = []

bad_epochs = []

adv_wait_times = []
root_decision_delay_times = []

root_selection_times = []
dstwr_comm_times_first_tx = []
dstwr_comm_times_last_tx = []
positioning_times = []
tof_times = []
position_after_tof_times = []
total_times = []
leaf_collection_times = []
leaf_to_dstwr_delay_times = []
leaf_to_dstwr_total_times = []
dstwr_msg_diff_values = []

bad_epochs = []

for idx, e in enumerate(epochs):
    adv_first = e["adv_first"]
    adv_last = e["adv_last"]
    root_selected = e["root_selected"]
    tx_first = e["tx_first"]
    tx_last = e["tx_last"]
    dstwr = e["dstwr"]
    tof = e["tof"]
    position = e["position"]
    leaf_first = e["leaf_first"]
    leaf_last = e["leaf_last"]

    # ADV wait time: 첫 ADV -> 마지막 ADV
    if adv_first is not None and adv_last is not None:
        v = diff_ms(adv_first, adv_last)
        adv_wait_times.append(v)

    # Root decision delay: 마지막 ADV -> ROOT SELECTED
    if adv_last is not None and root_selected is not None:
        v = diff_ms(adv_last, root_selected)
        root_decision_delay_times.append(v)

    # 기존 전체 root selection 구간: 첫 ADV -> ROOT SELECTED
    if adv_first is not None and root_selected is not None:
        v = diff_ms(adv_first, root_selected)
        root_selection_times.append(v)

    # first TX -> DS-TWR
    if tx_first is not None and dstwr is not None:
        v = diff_ms(tx_first, dstwr)
        dstwr_comm_times_first_tx.append(v)

    # last TX -> DS-TWR
    if tx_last is not None and dstwr is not None:
        v = diff_ms(tx_last, dstwr)
        dstwr_comm_times_last_tx.append(v)

    # DS-TWR -> POSITION
    if dstwr is not None and position is not None:
        v = diff_ms(dstwr, position)
        positioning_times.append(v)

    # DS-TWR -> TOF
    if dstwr is not None and tof is not None:
        v = diff_ms(dstwr, tof)
        tof_times.append(v)

    # TOF -> POSITION
    if tof is not None and position is not None:
        v = diff_ms(tof, position)
        position_after_tof_times.append(v)

    # first ADV -> POSITION
    if adv_first is not None and position is not None:
        v = diff_ms(adv_first, position)
        total_times.append(v)

    if adv_first is not None and root_selected is not None:
        root_ms = diff_ms(adv_first, root_selected)
        if root_ms > 1000:
            bad_epochs.append((idx, e["tag"], e["seq"], root_ms))

    if leaf_first is not None and leaf_last is not None:
        leaf_collection_times.append(diff_ms(leaf_first, leaf_last))

    if leaf_last is not None and dstwr is not None:
        leaf_to_dstwr_delay_times.append(diff_ms(leaf_last, dstwr))

    if leaf_first is not None and dstwr is not None:
        leaf_to_dstwr_total_times.append(diff_ms(leaf_first, dstwr))

    if e["dstwr_msg_diff_s"] is not None:
        dstwr_msg_diff_values.append(e["dstwr_msg_diff_s"])


print(f"Total epochs parsed: {len(epochs)}")
print(f"Bad root-selection epochs > 1000 ms: {len(bad_epochs)}")

print_summary("ADV wait time: first ADV -> last ADV", adv_wait_times)

print_summary("DS-TWR communication time: first TX -> DS-TWR", dstwr_comm_times_first_tx)
print_summary("DS-TWR communication time: last TX -> DS-TWR", dstwr_comm_times_last_tx)

if dstwr_msg_diff_values:
    avg_diff_s = mean(dstwr_msg_diff_values)
    print(f"\n[DS-TWR msg[1]+msg[0] * DWT_TIME_UNIT]")
    print(f"  n      = {len(dstwr_msg_diff_values)}")
    print(f"  avg    = {avg_diff_s:.9f} s  ({avg_diff_s*1000:.6f} ms)")
    print(f"  median = {median(dstwr_msg_diff_values):.9f} s")
    print(f"  min    = {min(dstwr_msg_diff_values):.9f} s")
    print(f"  max    = {max(dstwr_msg_diff_values):.9f} s")

if bad_epochs:
    print("\n[Bad epochs example]")
    for item in bad_epochs[:10]:
        idx, tag, seq, root_ms = item
        print(f"  epoch_idx={idx}, tag={tag}, seq={seq}, root_selection={root_ms:.3f} ms")