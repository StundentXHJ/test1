import uuid
import numpy as np

class Point:
    def __init__(self, x: int, y: int, step: int, clazz):
        self.x = x
        self.y = y
        self.step = step
        self.history = []
        self.id = str(uuid.uuid4())
        self.checked_cnt_ids = []
        self.history.append((x, y, step, 0))
        self.able = True
        self.clazz = clazz
        self.in_gate_name = None
        self.out_gate_name = None

    def add_history(self, x: int, y: int, step: int):
        self.x = x
        self.y = y
        self.step = step
        self.history.append((x, y, step, np.sqrt((self.x-self.history[-1][0])**2+(self.y-self.history[-1][1])**2)))

    def check_point(self, box):
        bx, by, bw, bh = box
        if bx < self.x < bx + bw and by < self.y < by + bh:
            return self.id
        else:
            return None

    def set_disable(self):
        self.able = False
