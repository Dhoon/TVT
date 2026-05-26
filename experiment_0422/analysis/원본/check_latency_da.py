import re
from statistics import mean, median

LOG_PATH = "log_20260422_221958_mode1.txt"


def parse_time_to_us(t: str) -> int:
    """
    Format: HH:MM:SS:mmm:uuu
    Example: 22:20:50:730:838
    Returns microseconds from start of day.
    """
    hh, mm, ss, ms, us = map(int, t.split(":"))
    return (((hh * 60 + mm) * 60 + ss) * 1_000_000) + ms * 1000 + us


def diff_ms(start_us: int, end_us: int) -> float:
    """
    Handles normal case and midnight wrap-around.
    """
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
    print(f"  avg    = {s['avg_ms']:.6f} ms")
    print(f"  median = {s['median_ms']:.6f} ms")
    print(f"  min    = {s['min_ms']:.6f} ms")
    print(f"  max    = {s['max_ms']:.6f} ms")


time_pattern = r"(\d{2}:\d{2}:\d{2}:\d{3}:\d{3})"

# Anchor 번호까지 파싱해야 selected root anchor의 ADV 시간을 찾을 수 있음
adv_re = re.compile(
    time_pattern
    + r".*\[ADV\]\s+From Anchor\s+(\d+),\s+Tag\s+(\d+),\s+Seq=(\d+)"
)

root_re = re.compile(
    time_pattern
    + r".*\[ROOT SELECTED\]\s+Anchor\s+(\d+)\s+for\s+Tag\s+(\d+)\s+Seq\s+(\d+)"
)

tx_re = re.compile(time_pattern + r".*\[TX\]\s+Sent root ID\s+(\d+)")

leaf_re = re.compile(
    time_pattern
    + r".*\[Leaf\].*Anchor\s+(\d+),\s+Tag\s+(\d+),\s+Seq=(\d+)"
)

dstwr_re = re.compile(
    time_pattern
    + r".*\[DS-TWR\].*Anchor\s+(\d+),\s+Tag\s+(\d+),\s+Seq=(\d+)"
)

tof_re = re.compile(time_pattern + r".*\[TOF\]")
pos_re = re.compile(time_pattern + r".*\[POSITION\]")


def new_epoch(tag, seq, adv_time, adv_anchor):
    return {
        "tag": int(tag),
        "seq": int(seq),

        # ADV timing
        "adv_first": adv_time,
        "adv_last": adv_time,

        # 핵심 추가:
        # anchor별 ADV 도착 시간 저장
        "adv_times_by_anchor": {
            int(adv_anchor): adv_time
        },

        # ROOT SELECTED
        "root_selected": None,
        "root_anchor": None,

        # TX / Leaf / DS-TWR / TOF / Position
        "tx_first": None,
        "tx_last": None,
        "leaf_first": None,
        "leaf_last": None,
        "dstwr": None,
        "tof": None,
        "position": None,
    }


epochs = []
current_epoch = None
last_dstwr_epoch = None

with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        # ADV
        m = adv_re.match(line)
        if m:
            t, adv_anchor, tag, seq = m.groups()

            ts = parse_time_to_us(t)
            adv_anchor = int(adv_anchor)
            tag = int(tag)
            seq = int(seq)

            # 새 epoch 조건:
            # 1) 현재 epoch 없음
            # 2) seq/tag 변경
            # 3) 이미 ROOT SELECTED 이후에 ADV가 다시 등장
            if (
                current_epoch is None
                or current_epoch["seq"] != seq
                or current_epoch["tag"] != tag
                or current_epoch["root_selected"] is not None
            ):
                current_epoch = new_epoch(tag, seq, ts, adv_anchor)
                epochs.append(current_epoch)
            else:
                # 같은 Seq의 여러 ADV 수신
                current_epoch["adv_last"] = ts
                current_epoch["adv_times_by_anchor"][adv_anchor] = ts

            continue

        # ROOT SELECTED
        m = root_re.match(line)
        if m:
            t, root, tag, seq = m.groups()

            ts = parse_time_to_us(t)
            root = int(root)
            tag = int(tag)
            seq = int(seq)

            if (
                current_epoch is not None
                and current_epoch["tag"] == tag
                and current_epoch["seq"] == seq
            ):
                current_epoch["root_selected"] = ts
                current_epoch["root_anchor"] = root

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
            t, anchor, tag, seq = m.groups()

            ts = parse_time_to_us(t)
            tag = int(tag)
            seq = int(seq)

            if (
                current_epoch is not None
                and current_epoch["tag"] == tag
                and current_epoch["seq"] == seq
            ):
                current_epoch["dstwr"] = ts
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


# =========================
# Metric containers
# =========================

# 네가 원하는 핵심 지표
selected_root_adv_arrival_times = []
selected_root_adv_missing_count = 0

# 기존 지표들
adv_wait_times = []
root_decision_delay_times = []
root_selection_total_times = []

dstwr_comm_times_first_tx = []
dstwr_comm_times_last_tx = []

leaf_collection_times = []
leaf_to_dstwr_delay_times = []
leaf_to_dstwr_total_times = []

positioning_times = []
tof_times = []
position_after_tof_times = []
total_times = []

bad_epochs = []

root_selected_anchor_counts = {i: 0 for i in range(1, 7)}


for idx, e in enumerate(epochs):
    adv_first = e["adv_first"]
    adv_last = e["adv_last"]
    adv_times_by_anchor = e["adv_times_by_anchor"]

    root_selected = e["root_selected"]
    root_anchor = e["root_anchor"]

    tx_first = e["tx_first"]
    tx_last = e["tx_last"]

    leaf_first = e["leaf_first"]
    leaf_last = e["leaf_last"]

    dstwr = e["dstwr"]
    tof = e["tof"]
    position = e["position"]

    # =====================================================
    # 핵심 지표:
    # 첫 ADV -> 최종 선택된 root anchor의 ADV가 들어온 시간
    # 예: first ADV Anchor 2, selected root Anchor 3
    #     Anchor 2 ADV time -> Anchor 3 ADV time
    # =====================================================
    if adv_first is not None and root_anchor is not None:
        root_adv_time = adv_times_by_anchor.get(root_anchor)

        if root_adv_time is not None:
            v = diff_ms(adv_first, root_adv_time)

            # 선택된 root anchor가 첫 ADV로 들어온 경우
            # 0.000 ms 대신 0.001 ms로 처리
            if v == 0:
                v = 0.001

            selected_root_adv_arrival_times.append(v)
        else:
            # ROOT SELECTED된 anchor의 ADV 로그가 해당 epoch에 없는 경우
            selected_root_adv_missing_count += 1

    # root anchor 선택 분포
    if root_anchor is not None:
        root_selected_anchor_counts[root_anchor] = (
            root_selected_anchor_counts.get(root_anchor, 0) + 1
        )

    # ADV wait time: 첫 ADV -> 마지막 ADV
    if adv_first is not None and adv_last is not None:
        adv_wait_times.append(diff_ms(adv_first, adv_last))

    # Root decision delay: 마지막 ADV -> ROOT SELECTED
    if adv_last is not None and root_selected is not None:
        root_decision_delay_times.append(diff_ms(adv_last, root_selected))

    # Root selection total time: 첫 ADV -> ROOT SELECTED
    if adv_first is not None and root_selected is not None:
        root_ms = diff_ms(adv_first, root_selected)
        root_selection_total_times.append(root_ms)

        if root_ms > 1000:
            bad_epochs.append((idx, e["tag"], e["seq"], root_ms))

    # first TX -> DS-TWR
    if tx_first is not None and dstwr is not None:
        dstwr_comm_times_first_tx.append(diff_ms(tx_first, dstwr))

    # last TX -> DS-TWR
    if tx_last is not None and dstwr is not None:
        dstwr_comm_times_last_tx.append(diff_ms(tx_last, dstwr))

    # Leaf collection: first Leaf -> last Leaf
    if leaf_first is not None and leaf_last is not None:
        leaf_collection_times.append(diff_ms(leaf_first, leaf_last))

    # last Leaf -> DS-TWR
    if leaf_last is not None and dstwr is not None:
        leaf_to_dstwr_delay_times.append(diff_ms(leaf_last, dstwr))

    # first Leaf -> DS-TWR
    if leaf_first is not None and dstwr is not None:
        leaf_to_dstwr_total_times.append(diff_ms(leaf_first, dstwr))

    # DS-TWR -> POSITION
    if dstwr is not None and position is not None:
        positioning_times.append(diff_ms(dstwr, position))

    # DS-TWR -> TOF
    if dstwr is not None and tof is not None:
        tof_times.append(diff_ms(dstwr, tof))

    # TOF -> POSITION
    if tof is not None and position is not None:
        position_after_tof_times.append(diff_ms(tof, position))

    # first ADV -> POSITION
    if adv_first is not None and position is not None:
        total_times.append(diff_ms(adv_first, position))


# =========================
# Print results
# =========================

print(f"Total epochs parsed: {len(epochs)}")
print(f"Bad root-selection epochs > 1000 ms: {len(bad_epochs)}")

print_summary(
    "Selected root ADV arrival time: first ADV -> selected root anchor ADV",
    selected_root_adv_arrival_times
)
print(f"\nSelected root ADV missing count: {selected_root_adv_missing_count}")

print_summary("ADV wait time: first ADV -> last ADV", adv_wait_times)
print_summary("Root decision delay: last ADV -> ROOT SELECTED", root_decision_delay_times)
print_summary("Root selection total time: first ADV -> ROOT SELECTED", root_selection_total_times)

print_summary("DS-TWR communication time: first TX -> DS-TWR", dstwr_comm_times_first_tx)
print_summary("DS-TWR communication time: last TX -> DS-TWR", dstwr_comm_times_last_tx)

print_summary("Leaf collection time: first Leaf -> last Leaf", leaf_collection_times)
print_summary("Leaf-to-DS-TWR delay: last Leaf -> DS-TWR", leaf_to_dstwr_delay_times)
print_summary("Leaf-to-DS-TWR total time: first Leaf -> DS-TWR", leaf_to_dstwr_total_times)

print_summary("Positioning time: DS-TWR -> POSITION", positioning_times)
print_summary("TOF calculation time: DS-TWR -> TOF", tof_times)
print_summary("Position calculation time: TOF -> POSITION", position_after_tof_times)
print_summary("Total time: first ADV -> POSITION", total_times)

print("\n[Root selected anchor distribution]")
total_root_selected = sum(root_selected_anchor_counts.values())
for anchor in range(1, 7):
    count = root_selected_anchor_counts.get(anchor, 0)
    ratio = count / total_root_selected * 100 if total_root_selected > 0 else 0
    print(f"  Anchor {anchor}: {count:4d} ({ratio:5.1f}%)")

if bad_epochs:
    print("\n[Bad epochs example]")
    for item in bad_epochs[:10]:
        idx, tag, seq, root_ms = item
        print(
            f"  epoch_idx={idx}, tag={tag}, seq={seq}, "
            f"root_selection={root_ms:.3f} ms"
        )