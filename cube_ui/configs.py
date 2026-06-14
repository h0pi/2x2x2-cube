import numpy as np

rubiks_moves = {
    # Level-1 faces (U, R, F): positive angle matches logical rotation direction.
    'U':  (np.radians( 90.),  np.array([0, 1, 0]), 1),
    "U'": (np.radians(-90.),  np.array([0, 1, 0]), 1),

    'R':  (np.radians( 90.),  np.array([1, 0, 0]), 1),
    "R'": (np.radians(-90.),  np.array([1, 0, 0]), 1),

    'F':  (np.radians( 90.),  np.array([0, 0, 1]), 1),
    "F'": (np.radians(-90.),  np.array([0, 0, 1]), 1),

    # Level-0 faces (D, L, B): angle must be negated because these layers are
    # on the negative side of their axis — same rotation matrix produces the
    # opposite effect relative to the logical move table.
    'D':  (np.radians(-90.),  np.array([0, 1, 0]), 0),
    "D'": (np.radians( 90.),  np.array([0, 1, 0]), 0),

    'L':  (np.radians(-90.),  np.array([1, 0, 0]), 0),
    "L'": (np.radians( 90.),  np.array([1, 0, 0]), 0),

    'B':  (np.radians(-90.),  np.array([0, 0, 1]), 0),
    "B'": (np.radians( 90.),  np.array([0, 0, 1]), 0),
}

window_w = 900
window_h = 900
fps = 60

# camera is created inside main() after init_window() — do not define here
