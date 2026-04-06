import threading

import anchor_selection
import state
from logger import log, setup_logger, timestamp
from serial_manager import monitor_ports
from ui import run_ui

if __name__ == "__main__":
    setup_logger()
    while True:
        try:
            mode = int(input("Root anchor selection mode (1: round-robin, 2: max power): "))
            if mode in (1, 2):
                anchor_selection.selection_mode = mode
                break
        except ValueError:
            pass
        print("1 또는 2를 입력하세요.")
    log(f"{timestamp()} Monitoring new COM ports...")
    t = threading.Thread(target=monitor_ports, daemon=True)
    t.start()
    run_ui()
    log(f"{timestamp()} Stopping.")
    for ser in state.serial_connections.values():
        ser.close()
