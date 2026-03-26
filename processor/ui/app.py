import queue
import tkinter as tk

import state
from settings import ANCHOR_POSITIONS

CANVAS_W = 550
CANVAS_H = 500
MARGIN = 60


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("UWB Positioning Monitor")
        self.root.configure(bg='#1e1e1e')

        self._build_layout()
        self._setup_canvas()
        self._poll()

    def _build_layout(self):
        left = tk.Frame(self.root, bg='#1e1e1e')
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=10, pady=10)

        header = tk.Frame(left, bg='#1e1e1e')
        header.pack(fill=tk.X)
        tk.Label(header, text="Anchor Map", bg='#1e1e1e', fg='#aaaaaa',
                 font=('Consolas', 11, 'bold')).pack(side=tk.LEFT)
        self.anchor_count_label = tk.Label(header, text="Anchors: 0 / 6",
                 bg='#1e1e1e', fg='#e05555', font=('Consolas', 10, 'bold'))
        self.anchor_count_label.pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(left, width=CANVAS_W, height=CANVAS_H,
                                bg='#2d2d2d', highlightthickness=1,
                                highlightbackground='#555555')
        self.canvas.pack()

    def _setup_canvas(self):
        positions = list(ANCHOR_POSITIONS.values())
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        self._x_min = min(xs) - 31
        self._x_max = max(xs) + 31
        self._y_min = min(ys) - 31
        self._y_max = max(ys) + 31

        self._sx = (CANVAS_W - 2 * MARGIN) / (self._x_max - self._x_min)
        self._sy = (CANVAS_H - 2 * MARGIN) / (self._y_max - self._y_min)

        for i in range(int(self._x_min), int(self._x_max) + 2):
            x, _ = self._to_canvas(i, self._y_min)
            self.canvas.create_line(x, MARGIN, x, CANVAS_H - MARGIN,
                                    fill='#3a3a3a', dash=(2, 4))
        for j in range(int(self._y_min), int(self._y_max) + 2):
            _, y = self._to_canvas(self._x_min, j)
            self.canvas.create_line(MARGIN, y, CANVAS_W - MARGIN, y,
                                    fill='#3a3a3a', dash=(2, 4))

        for anchor_id, (ax, ay) in ANCHOR_POSITIONS.items():
            cx, cy = self._to_canvas(ax, ay)
            r = 14
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    fill='#4a90d9', outline='#aaccff', width=1)
            self.canvas.create_text(cx, cy, text=str(anchor_id),
                                    fill='white', font=('Consolas', 9, 'bold'))
            self.canvas.create_text(cx, cy + r + 10,
                                    text=f"({ax},{ay})",
                                    fill='#888888', font=('Consolas', 7))

        self._tag_label = None
        self._tag_oval = None

    def _to_canvas(self, x, y):
        cx = MARGIN + (x - self._x_min) * self._sx
        cy = CANVAS_H - MARGIN - (y - self._y_min) * self._sy
        return cx, cy

    def _update_tag(self, x, y):
        if self._tag_oval:
            self.canvas.delete(self._tag_oval)
        if self._tag_label:
            self.canvas.delete(self._tag_label)
        cx, cy = self._to_canvas(x, y)
        r = 10
        self._tag_oval = self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill='#e05555', outline='#ff9999', width=2)
        self._tag_label = self.canvas.create_text(
            cx, cy - r - 10,
            text=f"({x:.2f}, {y:.2f})",
            fill='#ff9999', font=('Consolas', 8, 'bold'))

    def _poll(self):
        try:
            while True:
                msg = state.ui_queue.get_nowait()
                if msg['type'] == 'position':
                    self._update_tag(msg['x'], msg['y'])
        except queue.Empty:
            pass
        count = len(state.serial_connections)
        color = '#44cc66' if count >= state.MAX_ANCHORS else '#e05555'
        self.anchor_count_label.configure(
            text=f"Anchors: {count} / {state.MAX_ANCHORS}", fg=color)
        self.root.after(100, self._poll)


def run_ui():
    root = tk.Tk()
    App(root)
    root.mainloop()
