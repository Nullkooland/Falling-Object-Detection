from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json

# plt.style.use('science')

# Define parameters
NUM_BOXES = 8
NUM_FRAMES = 256
INIT_WIDTH = 48
INIT_HEIGH = 48

SIDE_MIN = 16
SIDE_MAX = 128

MARGIN = 0
X_MAX = 1280
Y_MAX = 720

STD_ACC_POS = 0.5
STD_ACC_SIZE = 1e-1
dt = 1

if __name__ == "__main__":
    bbox = np.zeros((NUM_BOXES, 4))
    velocity = np.zeros((NUM_BOXES, 4))

    bbox_tracks_gt = np.empty((NUM_FRAMES, NUM_BOXES, 4))

    patches = []
    cmap = plt.get_cmap("tab10")

    for i in range(NUM_BOXES):
        x = np.random.uniform(0, X_MAX - INIT_WIDTH)
        y = np.random.uniform(0, Y_MAX - INIT_HEIGH)
        bbox[i] = np.array([x, y, INIT_WIDTH, INIT_HEIGH])

    for t in range(NUM_FRAMES):
        for i in range(NUM_BOXES):
            bbox[i, 0] += velocity[i, 0] * dt
            bbox[i, 1] += velocity[i, 1] * dt
            bbox[i, 2] += velocity[i, 2] * dt
            bbox[i, 3] += velocity[i, 3] * dt

            if (bbox[i, 0] < MARGIN and velocity[i, 0] < 0.0) or (bbox[i, 0] + bbox[i, 2] > X_MAX - MARGIN and velocity[i, 0] > 0.0):
                velocity[i, 0] = -velocity[i, 0] * 0.5

            if (bbox[i, 1] < MARGIN and velocity[i, 1] < 0.0) or (bbox[i, 1] + bbox[i, 3] > Y_MAX - MARGIN and velocity[i, 1] > 0.0):
                velocity[i, 1] = -velocity[i, 1] * 0.5

            if (bbox[i, 2] < SIDE_MIN and velocity[i, 2] < 0.0) or (bbox[i, 2] > SIDE_MAX and velocity[i, 2] > 0.0):
                velocity[i, 2] = -velocity[i, 2] * 0.5

            if (bbox[i, 3] < SIDE_MIN and velocity[i, 3] < 0.0) or (bbox[i, 3] > SIDE_MAX and velocity[i, 3] > 0.0):
                velocity[i, 3] = -velocity[i, 3] * 0.5

            velocity[i, :2] += np.random.randn(2) * STD_ACC_POS
            velocity[i, 2:] += np.random.randn(2) * STD_ACC_SIZE

            bbox_tracks_gt[t, i] = bbox[i]

    gen_bboxes_obj  = {
        'type-id': 'opencv-matrix',
        'sizes': bbox_tracks_gt.shape,
        'dt': 'f',
        'data': bbox_tracks_gt.flatten().tolist()
    }

    with open('data/gen_bboxes.json', 'w') as f:
        json.dump({"gen_bboxes" : gen_bboxes_obj} , f)


    for i in range(NUM_BOXES):
        x, y, width, height = bbox_tracks_gt[0, i]
        patch = plt.Rectangle((x, y), width, height, color=cmap(i))
        patches.append(patch)

    fig, ax = plt.subplots(1, 1, num='Random bbox walk', figsize=(12, 8))
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, Y_MAX)
    ax.set_aspect('equal')
    ax.grid()

    def init():
        for patch in patches:
            ax.add_patch(patch)
        return patches

    def animate(t):
        for i, patch in enumerate(patches):
            x, y, width, height = bbox_tracks_gt[t, i]
            patch.set_xy((x, y))
            patch.set_width(width)
            patch.set_height(height)

        return patches

    anim = FuncAnimation(fig, animate,
                         init_func=init,
                         frames=range(NUM_FRAMES),
                         interval=20,
                         blit=True)

    plt.show()
