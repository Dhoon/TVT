import re
import threading
import time

import serial
import serial.tools.list_ports

import state
from logger import log, timestamp
from message_handler import handle_message


def read_uart_loop(port, ser):
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
                        message = buffer[start + 1:end]
                        handle_message(port, message)
                        buffer = buffer[end + 1:]
                    else:
                        break
        except Exception as e:
            log(f"{timestamp()} [{port}] read error: {e}")
            ser.close()
            state.serial_connections.pop(port, None)
            state.device_roles.pop(port, None)
            state.anchor_change_event.set()
            return


def identify_device(ser, port):
    buffer = ""
    start_time = time.time()
    while time.time() - start_time < 10.0:
        if ser.in_waiting:
            data = ser.read(ser.in_waiting).decode(errors='ignore')
            buffer += data
            log(f"{timestamp()} [{port}] raw: {repr(data)}")
            match = re.search(r"ROLE=([A-Z]+),\s*ID=([A-Z0-9]+)", buffer)
            if match:
                return match.group(2)
        time.sleep(0.05)
    return None


def handle_new_port(port):
    if port in state.serial_connections or port in state.failed_ports:
        return
    try:
        ser = serial.Serial(port, 115200, timeout=0.3, dsrdtr=False, rtscts=False)
        log(f"{timestamp()} [{port}] opened.")
        role = identify_device(ser, port)
        if role:
            state.device_roles[port] = role
            state.serial_connections[port] = ser
            log(f"{timestamp()} [{port}] identified as {role}")
            threading.Thread(target=read_uart_loop, args=(port, ser), daemon=True).start()
        else:
            ser.close()
            if port in state.known_ports and port not in state.serial_connections:
                log(f"{timestamp()} [{port}] no role message. closing.")
                state.failed_ports.add(port)
    except Exception as e:
        log(f"{timestamp()} [{port}] open error: {e}")
        state.known_ports.discard(port)


def handle_removed_port(port):
    ser = state.serial_connections.pop(port, None)
    state.device_roles.pop(port, None)
    state.failed_ports.discard(port)
    if ser:
        try:
            ser.close()
        except Exception:
            pass
    log(f"{timestamp()} [{port}] disconnected. removed.")


def monitor_ports():
    while True:
        if len(state.serial_connections) >= state.MAX_ANCHORS:
            state.anchor_change_event.wait()
            state.anchor_change_event.clear()

        current_ports = set(
            p.device for p in serial.tools.list_ports.comports()
            if p.vid is not None or 'USB' in (p.device or '') or 'ACM' in (p.device or '')
        )
        new_ports = current_ports - state.known_ports
        removed_ports = state.known_ports - current_ports
        for port in new_ports:
            threading.Thread(target=handle_new_port, args=(port,), daemon=True).start()
        for port in removed_ports:
            handle_removed_port(port)
        state.known_ports = current_ports
        time.sleep(0.3)
