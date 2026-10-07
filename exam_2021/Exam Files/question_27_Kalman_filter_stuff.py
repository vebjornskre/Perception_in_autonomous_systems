import numpy as np

def kf_predict_constant_velocity(x, P, dt=1.0, Q=None):
    """
    Kalman Filter prediction step for a 4-state constant-velocity model:
    state = [x, xdot, y, ydot].
    """

    F = np.array([
        [1, dt, 0,  0],
        [0, 1,  0,  0],
        [0, 0,  1, dt],
        [0, 0,  0,  1]
    ])

    if Q is None:
        Q = np.zeros((4, 4))

    x_pred = F @ x
    P_pred = F @ P @ F.T + Q

    return x_pred, P_pred


def kf_update_only(x, P, z, R):
    """
    Only the KF update step using a 2D measurement [x_meas, y_meas].
    """

    H = np.array([
        [1, 0, 0, 0],   # measure x
        [0, 0, 1, 0]    # measure y
    ])

    y_residual = z - H @ x                # innovation
    S = H @ P @ H.T + R                   # innovation covariance
    K = P @ H.T @ np.linalg.inv(S)        # Kalman gain

    x_upd = x + K @ y_residual
    I = np.eye(4)
    P_upd = (I - K @ H) @ P

    return x_upd, P_upd


# ---------------------------------------------------------------------
# Example usage: Task 1 + Task 2
# ---------------------------------------------------------------------
if __name__ == "__main__":

    print('####### TASK 1 — Prediction ########')

    # Prior state
    x = np.array([3.0, 0.5, 2.0, 0.33])

    # Prior covariance
    P = np.array([
        [5, 1, 0, 0],
        [1, 2, 0, 0],
        [0, 0, 5, 1],
        [0, 0, 1, 2]
    ])

    dt = 1.0

    x_pred, P_pred = kf_predict_constant_velocity(x, P, dt)

    print("Predicted state:")
    print(x_pred)
    print("\nPredicted covariance:")
    print(P_pred)


    print('\n####### TASK 2 — Update ########')

    x = np.array([5.0, 0.5, 7.0, 0.8])

    P = np.array([
        [0.2, 0, 0, 0],
        [0.2, 0.1, 0, 0],
        [0, 0, 0.2, 0],
        [0, 0, 0.2, 0.1]
    ])

    # Measurement (x, y)
    z = np.array([4.8, 7.1])

    # 2x2 measurement noise
    R = np.array([
        [0.2, 0.2],
        [0.2, 0.2]
    ])

    # IMPORTANT: update must use predicted state, not the prior
    x_upd, P_upd = kf_update_only(x, P, z, R)

    print("\nUpdated state:")
    print(x_upd)
