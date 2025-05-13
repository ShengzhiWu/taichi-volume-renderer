import numpy as np
import taichi as ti

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])

def rotation_matrix_to_quaternion(m):
    tr = np.trace(m)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    else:
        # Find the axis corresponding to the largest diagonal element
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / S
            x = 0.25 * S
            y = (m[0, 1] + m[1, 0]) / S
            z = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / S
            x = (m[0, 1] + m[1, 0]) / S
            y = 0.25 * S
            z = (m[1, 2] + m[2, 1]) / S
        else:
            S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / S
            x = (m[0, 2] + m[2, 0]) / S
            y = (m[1, 2] + m[2, 1]) / S
            z = 0.25 * S
    
    q = np.array([w, x, y, z])
    if w < 0:
        q = -q
    
    return q / np.linalg.norm(q)  # Normalize

def rotation_quaternion_to_matrix(q):
    return np.array([
        [1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] - q[3] * q[0]), 2 * (q[1] * q[3] + q[2] * q[0])],
        [2 * (q[1] * q[2] + q[3] * q[0]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] - q[1] * q[0])],
        [2 * (q[1] * q[3] - q[2] * q[0]), 2 * (q[2] * q[3] + q[1] * q[0]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)]
    ], dtype=float)

@ti.func
def rotation_quaternion_to_matrix_taichi(q):
    return ti.Matrix([
        [1 - 2 * (q.z ** 2 + q.w ** 2), 2 * (q.y * q.z - q.w * q.x), 2 * (q.y * q.w + q.z * q.x)],
        [2 * (q.y * q.z + q.w * q.x), 1 - 2 * (q.y ** 2 + q.w ** 2), 2 * (q.z * q.w - q.y * q.x)],
        [2 * (q.y * q.w - q.z * q.x), 2 * (q.z * q.w + q.y * q.x), 1 - 2 * (q.y ** 2 + q.z ** 2)]
    ])

@ti.func
def compute_covariance_inv(rotation, scale):  # type: ignore
    R = rotation_quaternion_to_matrix_taichi(rotation)
    
    S_squared_inv = ti.Matrix([
        [1. / scale.x ** 2, 0, 0],
        [0, 1. / scale.y ** 2, 0],
        [0, 0, 1. / scale.z ** 2]
    ])
    
    # Σ = RSSᵀRᵀ
    # Here is Σ^{-1}
    return R @ S_squared_inv @ R.transpose()
