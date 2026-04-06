import threading
import time

import state
from logger import log, timestamp
from positioning import estimate_tag_position
from anchor_selection import select_root_anchor

def handle_message(port, message):
    values = parse_numeric_list(message)
    if not values:
        log(f"{timestamp()} [{port}] Failed to parse message: {message}")
        return
    msg_type = values[0]
    if msg_type == 0:
        handle_error_message(port, values)
    elif msg_type == 12:
        handle_adv_message(port, values)
    elif msg_type in (23, 24):
        handle_ds_twr_message(port, values)


def handle_error_message(port, values):
    anchor_id = values[1]
    tag_id = values[2]
    sequence = values[3]
    error_state = values[4]
    log(f"{timestamp()} [ERROR] [{port}] Anchor {anchor_id} timeout, Tag {tag_id}, Seq={sequence}, State={error_state}. Clearing buffers.")

    key = (tag_id, sequence)
    with state.adv_lock:
        state.adv_messages.pop(key, None)
    with state.buffer_lock:
        state.message_buffer.pop(key, None)

    for p, ser in state.serial_connections.items():
        try:
            ser.write(f"[0]\r\n".encode())
            log(f"{timestamp()} [TX] Sent reset [0] to {p}")
        except Exception as e:
            log(f"{timestamp()} [TX ERROR] {p}: {e}")


def parse_numeric_list(message: str):
    try:
        cleaned = message.strip('[]').strip()
        if not cleaned:
            return []
        parts = cleaned.split(',')
        values = []
        for p in parts:
            num = p.strip()
            try:
                f = float(num)
                values.append(int(f) if f.is_integer() else f)
            except ValueError:
                return []
        return values
    except Exception as e:
        log(f"{timestamp()} [PARSE ERROR] {e}")
        return []


def handle_adv_message(port, values):
    anchor_id = values[1]
    tag_id = values[2]
    sequence = values[3]
    power = values[4]
    cir_values = values[5:]
    key = (tag_id, sequence)
    log(f"{timestamp()} [ADV] From Anchor {anchor_id}, Tag {tag_id}, Seq={sequence}, Power={power}, CIR={cir_values}")

    with state.adv_lock:
        if key not in state.adv_messages:
            state.adv_messages[key] = []
            threading.Thread(target=adv_timeout_handler, args=(key,), daemon=True).start()
        state.adv_messages[key].append(values)
        if len(state.adv_messages[key]) >= 6:
            msgs = state.adv_messages.pop(key)
            root_id = select_root_anchor(msgs)
            log(f"{timestamp()} [ROOT SELECTED] Anchor {root_id} for Tag {tag_id} Seq {sequence} (ADV Complete)")
            state.ui_queue.put({'type': 'root', 'anchor_id': root_id})
            for p, ser in state.serial_connections.items():
                try:
                    ser.write(f"[{root_id}]\r\n".encode())
                    log(f"{timestamp()} [TX] Sent root ID {root_id} to {p}")
                except Exception as e:
                    log(f"{timestamp()} [TX ERROR] {p}: {e}")


def handle_ds_twr_message(port, values):
    msg_type = values[0]
    anchor_id = values[1]
    tag_id = values[2]
    sequence = values[3]
    if msg_type == 23:
        lat = values[4]
        lon = values[5]
        log(f"{timestamp()} [DS-TWR] Type {msg_type}, Anchor {anchor_id}, Tag {tag_id}, Seq={sequence}, lat = {lat}, lon = {lon}, message = {values[6:]}")
    else:
        log(f"{timestamp()} [Leaf] Type {msg_type}, Anchor {anchor_id}, Tag {tag_id}, Seq={sequence}, message = {values[4:]}")

    key = (tag_id, sequence)
    data = None
    with state.buffer_lock:
        if key not in state.message_buffer:
            state.message_buffer[key] = []
            threading.Thread(target=ds_twr_timeout_handler, args=(key,), daemon=True).start()
        state.message_buffer[key].append(values)
        if len(state.message_buffer[key]) >= 6:
            data = state.message_buffer.pop(key, [])
    if data:
        estimate_tag_position(data)


def adv_timeout_handler(key):
    time.sleep(state.adv_timeout)
    with state.adv_lock:
        messages = state.adv_messages.pop(key, [])
    if messages:
        tag_id, sequence = key
        root_id = select_root_anchor(messages)
        log(f"{timestamp()} [ROOT SELECTED] Anchor {root_id} for Tag {tag_id} Seq {sequence} (ADV Timeout or Full)")
        state.ui_queue.put({'type': 'root', 'anchor_id': root_id})
        for p, ser in state.serial_connections.items():
            try:
                ser.write(f"[{root_id}]\r\n".encode())
                log(f"{timestamp()} [TX] Sent root ID {root_id} to {p}")
            except Exception as e:
                log(f"{timestamp()} [TX ERROR] {p}: {e}")


def ds_twr_timeout_handler(key):
    time.sleep(state.buffer_timeout)
    with state.buffer_lock:
        data = state.message_buffer.pop(key, [])
    if data:
        try:
            estimate_tag_position(data)
        except Exception as e:
            log(f"{timestamp()} [POSITION ERROR] {e}")