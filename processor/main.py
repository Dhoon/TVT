import threading

import state
from logger import log, setup_logger, timestamp
from serial_manager import monitor_ports
from ui import run_ui

if __name__ == "__main__":
    setup_logger()
    log(f"{timestamp()} Monitoring new COM ports...")
    t = threading.Thread(target=monitor_ports, daemon=True)
    t.start()
    run_ui()
    log(f"{timestamp()} Stopping.")
    for ser in state.serial_connections.values():
        ser.close()
