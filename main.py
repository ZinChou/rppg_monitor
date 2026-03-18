import multiprocessing as mp

from Monitor import Monitor

if __name__ == "__main__":
    mp.freeze_support()
    monitor = Monitor(
        camera_id=0,
        buffer_seconds=10,
        pos_window_seconds=1.6,
        display_scale=1.0,
        target_fps=30.0,
    )
    monitor.run()
