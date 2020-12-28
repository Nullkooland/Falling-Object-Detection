import numpy as np
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":

    with open('data/kalman_test.json', 'r') as f:
        input = json.load(f)
        N = input['zt']['rows']

        zt = np.array(input['zt']['data']).reshape(N, 2)
        xt_gt = np.array(input['xt_gt']['data']).reshape(N, 4)
        xt_est = np.array(input['xt_est']['data']).reshape(N, 4)

    pos_x_gt = xt_gt[..., 0]
    vel_x_gt = xt_gt[..., 1]
    pos_y_gt = xt_gt[..., 2]
    vel_y_gt = xt_gt[..., 3]

    pos_x_est = xt_est[..., 0]
    vel_x_est = xt_est[..., 1]
    pos_y_est = xt_est[..., 2]
    vel_y_est = xt_est[..., 3]

    pos_x_measured = zt[..., 0]
    pos_y_measured = zt[..., 1]

    fig_gt, axs = plt.subplots(1, 3,
                               figsize=(12, 4))

    axs[0].plot(pos_x_gt, pos_y_gt, label='Groundtruth')
    axs[0].plot(pos_x_est, pos_y_est, label='Estimate')
    axs[0].scatter(pos_x_measured, pos_y_measured, label='Measurement',
                   marker='.', s=16, color='cadetblue', alpha=0.8)

    axs[0].set_title('Trajectory')
    axs[0].set_xlabel(r'$x$')
    axs[0].set_ylabel(r'$y$')

    axs[1].plot(pos_x_gt, vel_x_gt, label='Groundtruth')
    axs[1].plot(pos_x_est, vel_x_est, label='Estimate')
    axs[1].set_title('Phase diagram - X')
    axs[1].set_xlabel(r'$x$')
    axs[1].set_ylabel(r'$v_x$')

    axs[2].plot(pos_y_gt, vel_y_gt, label='Groundtruth')
    axs[2].plot(pos_y_est, vel_y_est, label='Estimate')
    axs[2].set_title('Phase diagram - Y')
    axs[2].set_xlabel(r'$y$')
    axs[2].set_ylabel(r'$v_y$')

    axs[0].legend(loc='upper right')
    axs[1].legend(loc='upper right')
    axs[2].legend(loc='upper right')

    plt.show()