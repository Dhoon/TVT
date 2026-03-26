import os
from datetime import datetime

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
