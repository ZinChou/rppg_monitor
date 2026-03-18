import time
from collections import deque
from model.POS import POS
import cv2
import numpy as np
from model.POS import POS
from Monitor import Monitor

if __name__ == "__main__":
    model = POS()
    monitor = Monitor(
        model=model,
        camera_id=0,
        buffer_seconds=10,
        pos_window_seconds=1.6,
        display_scale=1.0,
    )
    monitor.run()
