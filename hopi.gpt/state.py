import pyray as pr

COLOR_MAP = {
    'W': 0, 'Y': 1, 'B': 2,
    'G': 3, 'O': 4, 'R': 5
}

def get_state(logic):
    state = []
    for face in ['U', 'D', 'F', 'B', 'L', 'R']:
        for c in logic.faces[face]:
            state.append(COLOR_MAP[c])
    return tuple(state)
