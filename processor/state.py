import queue
import threading

serial_connections = {}
device_roles = {}
failed_ports = set()
known_ports = set()
anchor_change_event = threading.Event()
ui_queue = queue.Queue()

MAX_ANCHORS = 6

adv_messages = {}
adv_lock = threading.Lock()
adv_timeout = 0.1  # seconds

message_buffer = {}
buffer_lock = threading.Lock()
buffer_timeout = 0.1  # seconds
