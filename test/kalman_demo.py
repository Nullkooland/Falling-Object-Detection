import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
import scipy

N = 1024
T = 4
dt = T / N

if __name__ == "__main__":
    # Time sequence
    t = np.linspace(0, T, N, endpoint=True)
    # Init Kalman filter
    kf = KalmanFilter(dim_x=4, dim_z=2, dim_u=1)

    # State transition matrix
    F = np.array([
        [1, dt,  0,  0],
        [0,  1,  0,  0],
        [0,  0,  1, dt],
        [0,  0,  0,  1],
    ])

    # Control transition matrix
    B = np.array([0, 0, 0.5 * dt ** 2, dt]).reshape(4, 1)

    # Control variable
    u = -scipy.constants.g

    # Specify init state
    x_init = np.array([0.0, 10.0, 0.0, 10.0]).reshape(4, 1)

    # Ground truth state sequence
    xt_gt = np.empty((4, N))
    xt_gt[..., 0] = x_init.flatten()

    # Specify measurement matrix
    H = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ])

    # Specify measurement noise
    R = np.diag([4.0, 25.0])

    # Measured state sequence
    zt = np.empty((2, N))
    n_z = np.random.multivariate_normal([0, 0], R)
    zt[..., 0] = x_init.flatten()[[0, 2]] + n_z

    # Generate groundtruth and measurement sequence
    for i in range(1, N):
        x_pre = xt_gt[..., i - 1].reshape(4, 1)
        x = (F @ x_pre + B * u).flatten()
        xt_gt[..., i] = x

        n_z = np.random.multivariate_normal([0, 0], R)
        zt[..., i] = x[[0, 2]] + n_z

    # Init Kalman filter
    kf = KalmanFilter(dim_x=4, dim_z=2, dim_u=1)

    # Assign transition matrices
    kf.F = F
    kf.H = H
    kf.B = B

    # Assgin covariance matrices
    kf.Q = np.zeros_like(F)
    kf.R = R
    kf.P = np.diag([16.0, 16.0, 16.0, 16.0])

    # Assign init state
    kf.x = x_init + np.random.randn(4, 1) * 4

    # Run Kalman filter
    xt_est = np.empty_like(xt_gt)

    for i in range(N):
        kf.update(zt[..., i].reshape(2, 1))
        kf.predict(u)
        xt_est[..., i] = kf.x.flatten()

    pos_x_gt = xt_gt[0, ...]
    vel_x_gt = xt_gt[1, ...]
    pos_y_gt = xt_gt[2, ...]
    vel_y_gt = xt_gt[3, ...]

    pos_x_est = xt_est[0, ...]
    vel_x_est = xt_est[1, ...]
    pos_y_est = xt_est[2, ...]
    vel_y_est = xt_est[3, ...]

    pos_x_measured = zt[0, ...]
    pos_y_measured = zt[1, ...]

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
