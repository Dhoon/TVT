import state

root_anchor_counter = 0


def select_root_anchor():
    global root_anchor_counter
    num_anchors = len(state.serial_connections)
    if num_anchors == 0:
        return None
    root_id = (root_anchor_counter % num_anchors) + 1
    root_anchor_counter += 1
    return root_id
