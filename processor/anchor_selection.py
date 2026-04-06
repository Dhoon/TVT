import state

root_anchor_counter = 0
selection_mode = 1  # 1: round-robin, 2: max power


def select_root_anchor(adv_msgs=None):
    if selection_mode == 2 and adv_msgs:
        return _select_by_power(adv_msgs)
    return _select_by_roundrobin()


def _select_by_roundrobin():
    global root_anchor_counter
    num_anchors = len(state.serial_connections)
    if num_anchors == 0:
        return None
    root_id = (root_anchor_counter % num_anchors) + 1
    root_anchor_counter += 1
    return root_id


def _select_by_power(adv_msgs):
    # adv_msgs: list of values, values[1]=anchor_id, values[4]=power
    best = max(adv_msgs, key=lambda m: m[4])
    return best[1]
