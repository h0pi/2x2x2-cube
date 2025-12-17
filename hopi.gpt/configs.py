import pyray as pr
import numpy as np

rubiks_moves = {
    'U':  (np.radians(90.),  np.array([0, 1, 0]), 1),
    "U'": (np.radians(-90),  np.array([0, 1, 0]), 1),

    'D':  (np.radians(90.),  np.array([0, 1, 0]), 0),
    "D'": (np.radians(-90),  np.array([0, 1, 0]), 0),

    'L':  (np.radians(90.),  np.array([1, 0, 0]), 0),
    "L'": (np.radians(-90.), np.array([1, 0, 0]), 0),

    'R':  (np.radians(90.),  np.array([1, 0, 0]), 1),
    "R'": (np.radians(-90.), np.array([1, 0, 0]), 1),

    'F':  (np.radians(90.),  np.array([0, 0, 1]), 1),
    "F'": (np.radians(-90.), np.array([0, 0, 1]), 1),

    'B':  (np.radians(90.),  np.array([0, 0, 1]), 0),
    "B'": (np.radians(-90.), np.array([0, 0, 1]), 0)
}



window_w =600
window_h = 600

fps = 60

#camera
camera = pr.Camera3D(
    pr.Vector3(18.0, 16.0, 18.0),   # position
    pr.Vector3(0.0, 0.0, 0.0),     # target
    pr.Vector3(0.0, 1.0, 0.0),     # up
    45.0,                          # fov
    pr.CameraProjection.CAMERA_PERSPECTIVE
)