import multiprocessing as mp

from Monitor_DP import Monitor
# from Monitor_TD import Monitor

if __name__ == "__main__":
    mp.freeze_support()
    monitor = Monitor(
        camera_id=0,
        buffer_seconds=40,
        pos_window_seconds=1.6,
        display_scale=1.0,
        target_fps=30.0,
    )
    monitor.run()
