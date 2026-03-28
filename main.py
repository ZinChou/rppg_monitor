import multiprocessing as mp

from Monitor_DP import Monitor_DP
from Monitor_TD import Monitor_TD

if __name__ == "__main__":
    mp.freeze_support()
    model = 1 # 0: POS;  1: Model.

    if model == 0:
        monitor = Monitor_TD(
            camera_id=0,
            buffer_seconds=10,
            pos_window_seconds=1.6,
            display_scale=1.0,
            target_fps=30.0,
        )
    else:
        monitor = Monitor_DP(
            
            camera_id=0,
            buffer_seconds=10,
            pos_window_seconds=1.6,
            display_scale=1.0,
            target_fps=30.0,
        )
    monitor.run()
