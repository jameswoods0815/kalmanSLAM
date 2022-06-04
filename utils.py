import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
from pathlib import Path
import gc
import os


def visualize_trajectory_2d(pose,
                            landmarks=None,
                            initialized=None,
                            observed=None,
                            xlim=None,
                            ylim=None,
                            figsize=20,
                            save_fig_name=None,
                            show_start_end=True,
                            show_navigation=False,
                            show_ori=False):
    PATH_COLOR = '#ff4733'
    LANDMARK_COLOR = '#59b359'
    LANDMARK_OBSERVED_COLOR = '#0067ff'
    NAVIGATION_COLOR = '#3888ff'

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    n_pose = pose.shape[2]

    ax.plot(pose[0, 3, :], pose[1, 3, :], '-', color=PATH_COLOR, linewidth=2)
    if landmarks is not None:
        n_landmarks = landmarks.shape[0]
        initialized = initialized.copy() if initialized is not None else np.ones(n_landmarks, dtype=bool)
        observed = observed.copy() if observed is not None else np.zeros(n_landmarks, dtype=bool)

        initialized[observed] = False

        ax.scatter(landmarks[initialized, 0],
                   landmarks[initialized, 1],
                   s=10.0,
                   color=LANDMARK_COLOR,
                   marker='o',
                   linewidths=0.1)
        ax.scatter(landmarks[observed, 0],
                   landmarks[observed, 1],
                   s=10.0,
                   color=LANDMARK_OBSERVED_COLOR,
                   marker='o',
                   linewidths=0.1)

    if show_start_end:
        ax.scatter(pose[0, 3, 0], pose[1, 3, 0], color='red', marker='s', label="start")
        ax.scatter(pose[0, 3, -1], pose[1, 3, -1], color='blue', marker='o', label="end")

    if show_ori:
        select_ori_index = list(range(0, n_pose, max(int(n_pose / 10), 1)))
        yaw_list = []
        for i in select_ori_index:
            _, _, yaw = mat2euler(pose[:3, :3, i])
            yaw_list.append(yaw)
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx, dy = [dx, dy] / np.sqrt(dx**2 + dy**2)
        ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
            color="b",units="xy",width=1)

    if show_navigation:
        dx, dy = 0, 0
        for i in range(2, n_pose + 1):
            if dx**2 + dy**2 > 1:
                break
            else:
                dx = np.average(pose[0, 3, -1] - pose[0, 3, -i:-1])
                dy = np.average(pose[1, 3, -1] - pose[1, 3, -i:-1])

        x, y = pose[0, 3, -1], pose[1, 3, -1]
        plt.arrow(x,
                  y,
                  dx,
                  dy,
                  length_includes_head=True,
                  head_width=10,
                  head_starts_at_zero=True,
                  overhang=0.2,
                  zorder=999,
                  facecolor=NAVIGATION_COLOR,
                  edgecolor='black')

    plt.axis('equal')
    # plt.axis('off')
    ax.grid(False)
    # ax.legend()

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    if save_fig_name:
        Path(os.path.dirname(save_fig_name)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_fig_name, bbox_inches='tight')
        # Clear the current figure.
        plt.clf()
        # Closes all the figure windows.
        plt.close('all')
        plt.close(fig)
        del fig
        del ax
        gc.collect()
        return None, None
    else:
        plt.show(block=True)
        return fig, ax
