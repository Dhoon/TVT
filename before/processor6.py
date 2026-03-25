import serial
import serial.tools.list_ports
import threading
import time
import re
import os
import sys
from datetime import datetime
import numpy as np
from scipy.optimize import least_squares

serial_connections = {}
device_roles = {}
failed_ports = set()
known_ports = set()

adv_messages = {}  # key: (tag_id, sequence), value: list of ADV messages
adv_lock = threading.Lock()
adv_timeout = 0.1  # seconds

message_buffer = {}  # key: (tag_id, sequence), value: list of DS-TWR messages
buffer_lock = threading.Lock()
buffer_timeout = 0.1  # seconds

# Timestamp helper
log_file = None

def timestamp():
    now = datetime.now()
    return f"{now.hour:02}:{now.minute:02}:{now.second:02}:{now.microsecond // 1000:03}:{now.microsecond % 1000:03}"

def setup_logger():
    global log_file
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log_{now}.txt")
    log_file = open(log_path, 'w', encoding='utf-8')

def log(message):
    print(message)
    if log_file:
        log_file.write(message + '\n')
        log_file.flush()

# Empty root selector (to be implemented)
root_anchor_counter = 0

def select_root_anchor():
    global root_anchor_counter
    root_id = (root_anchor_counter % 6) + 1  # Rotate through 1 to 6
    root_anchor_counter += 1
    return root_id

# Empty position estimator (to be implemented)
def estimate_tag_position(data):

    # Example anchor positions (anchor_id: (x, y))
    anchor_positions = {
        1: (0.0, 0.0),
        2: (5.0, 0.0),
        3: (0.0, 5.0),
        4: (5.0, 5.0),
        5: (2.5, 2.5),
        6: (7.0, 2.5),
    }

    c = 299702547  # speed of light in m/s

    root_msg = next((m for m in data if m and m[0] == 23 and len(m) >= 11), None)
    if not root_msg:
        log(f"{timestamp()} [POSITION] No valid root anchor or TOF")
        return

    root_anchor_id = root_msg[1]
    tag_id = root_msg[2]
    sequence = root_msg[3]
    lat, lon = root_msg[4], root_msg[5]
    Ra, Da, Rb, Db, D2b = root_msg[6:11]

    numerator = Ra * Rb - Da * Db
    denominator = Ra + Rb + Da + Db
    dwt_time_unit = 1 / (499.2e6 * 128.0)
    tof_dtu = numerator / denominator if denominator != 0 else 0
    root_toa = tof_dtu * dwt_time_unit
    root_dist = root_toa * c
    log(f"{timestamp()} [TOF] Root TOF: {tof_dtu:.2f} DTU, Distance: {root_dist:.2f} m")

    leaf_infos = []

    for msg in data:
        if not msg or msg[0] != 24 or len(msg) < 8:
            continue
        anchor_id = msg[1]
        t2, t5, t8, t11 = msg[4:8]

        Rt1 = t5 - t2
        Rt2 = t8 - t5
        Rt3 = t11 - t8
        if any(ts < 0 for ts in (Ra, Da, Rb, Db, D2b, Rt1, Rt2, Rt3)):
            log(f"{timestamp()} [WARNING] Negative timestamp detected, skipping anchor {anchor_id}")
            continue
        denominator = 2 * Rt1 + 2 * Rt2
        numerator = Da * Rt1 - Rt2 * Ra + Rb * Rt1 - Rt2 * Db
        tdoa_dtu = numerator / denominator if denominator != 0 else 0
        tdoa = tdoa_dtu * dwt_time_unit * c

        leaf_infos.append((anchor_id, tdoa))

    
    if root_anchor_id not in anchor_positions:
        log(f"{timestamp()} [POSITION] Unknown root anchor ID {root_anchor_id}")
        return
    root_pos = anchor_positions[root_anchor_id]
    
    leaf_positions = []
    tdoa_deltas = []
    for leaf_id, tdoa in leaf_infos:
        pos = anchor_positions.get(leaf_id)
        if pos:
            leaf_positions.append(pos)
            tdoa_deltas.append(tdoa)
        else:
            log(f"{timestamp()} [POSITION] Unknown leaf anchor ID {leaf_id}, skipping.")

    def residuals(p):
        x, y = p
        res = [(np.linalg.norm([x - root_pos[0], y - root_pos[1]]) - root_dist)]
        for (lx, ly), delta_d in zip(leaf_positions, tdoa_deltas):
            d_leaf = np.linalg.norm([x - lx, y - ly])
            d_leaf_root = np.linalg.norm([lx - root_pos[0], ly - root_pos[1]])
            res.append((d_leaf - d_leaf_root - delta_d))
        return res

    result = least_squares(residuals, x0=(0, 0))
    est_x, est_y = result.x
    if not result.success:
        log(f"{timestamp()} [POSITION] Optimization failed: {result.message}")
        return
    log(f"{timestamp()} [POSITION] Estimated tag position: ({est_x:.2f}, {est_y:.2f})")

# Timer thread for ADV processing
def adv_timeout_handler(key):
    time.sleep(adv_timeout)
    with adv_lock:
        messages = adv_messages.pop(key, [])
    if messages:
        tag_id, sequence = key
        root_id = select_root_anchor()
        log(f"{timestamp()} [ROOT SELECTED] Anchor {root_id} for Tag {tag_id} Seq {sequence} (ADV Timeout or Full)")
        for p, ser in serial_connections.items():
            try:
                ser.write(f"[{root_id}]\r\n".encode())
                log(f"{timestamp()} [TX] Sent root ID {root_id} to {p}")
            except Exception as e:
                log(f"{timestamp()} [TX ERROR] {p}: {e}")

# Timer thread for DS-TWR processing
def ds_twr_timeout_handler(key):
    time.sleep(buffer_timeout)
    with buffer_lock:
        data = message_buffer.pop(key, [])
    if data:
        try:
            estimate_tag_position(data)
        except Exception as e:
            log(f"{timestamp()} [POSITION ERROR] {e}")

# ADV message handler
def handle_adv_message(port, values):
    anchor_id = values[1]
    tag_id = values[2]
    sequence = values[3]
    power = values[4]
    cir_values = values[5:]
    key = (tag_id, sequence)
    log(f"{timestamp()} [ADV] From Anchor {anchor_id}, Tag {tag_id}, Seq={sequence}, Power={power}, CIR={cir_values}")

    with adv_lock:
        if key not in adv_messages:
            adv_messages[key] = []
            threading.Thread(target=adv_timeout_handler, args=(key,), daemon=True).start()
        adv_messages[key].append(values)
        if len(adv_messages[key]) >= 6:
            adv_messages.pop(key)
            root_id = select_root_anchor()
            log(f"{timestamp()} [ROOT SELECTED] Anchor {root_id} for Tag {tag_id} Seq {sequence} (ADV Complete)")
            for p, ser in serial_connections.items():
                try:
                    ser.write(f"[{root_id}]\r\n".encode())
                    log(f"{timestamp()} [TX] Sent root ID {root_id} to {p}")
                except Exception as e:
                    log(f"{timestamp()} [TX ERROR] {p}: {e}")

# DS-TWR message handler
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
    
    with buffer_lock:
        if key not in message_buffer:
            message_buffer[key] = []
            threading.Thread(target=ds_twr_timeout_handler, args=(key,), daemon=True).start()
        message_buffer[key].append(values)
        if len(message_buffer[key]) >= 6:
            data = message_buffer.get(key, [])
            if not data:
                return
            estimate_tag_position(data)

# 메시지 파서를 안전하게 작성 (float 허용)
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
                if f.is_integer():
                    values.append(int(f))
                else:
                    values.append(f)
            except ValueError:
                return []
        return values
    except Exception as e:
        log(f"{timestamp()} [PARSE ERROR] {e}")
        return []

# 메시지 디스패처
def handle_message(port, message):
    values = parse_numeric_list(message)
    if not values:
        log(f"{timestamp()} [{port}] Failed to parse message: {message}")
        return
    msg_type = values[0]
    if msg_type == 12:
        handle_adv_message(port, values)
    elif msg_type in (23, 24):
        handle_ds_twr_message(port, values)

# 수신 스레드
def read_uart_loop(port, ser):
    retry_delay = 10
    buffer = ""
    while True:
        try:
            if ser.in_waiting:
                data = ser.read(ser.in_waiting).decode(errors='ignore')
                buffer += data
                while True:
                    start = buffer.find('[')
                    end = buffer.find(']', start)
                    if start != -1 and end != -1:
                        message = buffer[start+1:end]
                        handle_message(port, message)
                        buffer = buffer[end+1:]
                    else:
                        break
        except Exception as e:
            log(f"{timestamp()} [{port}] read error: {e}")
            ser.close()
            failed_ports.add(port)
            time.sleep(retry_delay)
            failed_ports.discard(port)

# 부팅 메시지에서 ROLE=... 추출
def identify_device(ser, port):
    buffer = ""
    start_time = time.time()

    while time.time() - start_time < 30.0:
        if ser.in_waiting:
            data = ser.read(ser.in_waiting).decode(errors='ignore')
            buffer += data
            log(f"{timestamp()} [{port}] raw: {repr(data)}")

            match = re.search(r"ROLE=([A-Z]+),\s*ID=([A-Z0-9]+)", buffer)
            if match:
                return match.group(2)
        time.sleep(0.05)
    return None

# 새 포트 처리
def handle_new_port(port):
    if port in serial_connections or port in failed_ports:
        return
    try:
        ser = serial.Serial(port, 115200, timeout=0.3)
        log(f"{timestamp()} [{port}] opened.")
        role = identify_device(ser, port)
        if role:
            device_roles[port] = role
            serial_connections[port] = ser
            log(f"{timestamp()} [{port}] identified as {role}")
            threading.Thread(target=read_uart_loop, args=(port, ser), daemon=True).start()
        else:
            log(f"{timestamp()} [{port}] no role message. closing.")
            failed_ports.add(port)
            ser.close()
    except Exception:
        failed_ports.add(port)

# 포트 감시 루프
def monitor_ports():
    global known_ports
    while True:
        current_ports = set(p.device for p in serial.tools.list_ports.comports())
        new_ports = current_ports - known_ports
        if new_ports:
            for port in new_ports:
                threading.Thread(target=handle_new_port, args=(port,), daemon=True).start()
        known_ports = current_ports
        time.sleep(1)

if __name__ == "__main__":
    setup_logger()
    log(f"{timestamp()} Monitoring new COM ports...")
    try:
        monitor_ports()
    except KeyboardInterrupt:
        log(f"{timestamp()} Stopping.")
        for ser in serial_connections.values():
            ser.close()
