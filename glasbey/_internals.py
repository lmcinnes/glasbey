# MIT License
# Leland McInnes, Sergey Alexandrov

import numpy as np
import numba


@numba.njit(
    [
        "f4[::1](f4[::1], f4[:,::1], f4[::1])",
        numba.types.Array(numba.float32, 1, "C", readonly=True)(
            numba.types.Array(numba.float32, 1, "C"),
            numba.types.Array(numba.float32, 2, "C", readonly=True),
            numba.types.Array(numba.float32, 1, "C", readonly=True),
        ),
    ],
    locals={
        "diff": numba.float32,
        "d": numba.float32,
        "dim": numba.intp,
        "i": numba.uint32,
    },
    fastmath=True,
    nogil=True,
    boundscheck=False,
)
def get_next_color(distances, colors, new_color):
    argmax = -1
    max_dist = 0.0
    dim = distances.shape[0]
    for i in range(dim):
        d = 0.0
        for j in range(3):
            diff = colors[i, j] - new_color[j]
            d += diff * diff
        if d < distances[i]:
            distances[i] = d
        if distances[i] > max_dist:
            max_dist = distances[i]
            argmax = i

    return colors[argmax]


@numba.njit(
    [
        "f4[:,::1](f4[:,::1], f4[:,::1], i4)",
        numba.types.Array(numba.float32, 2, "C")(
            numba.types.Array(numba.float32, 2, "C", readonly=True),
            numba.types.Array(numba.float32, 2, "C", readonly=True),
            numba.uint32,
        ),
    ],
    locals={
        "distances": numba.types.Array(numba.float32, 1, "C"),
        "initial_palette_size": numba.uint16,
        "i": numba.uint16,
    },
)
def generate_palette_cam02ucs(colors, initial_palette, size):
    distances = np.full(colors.shape[0], fill_value=1e12, dtype=np.float32)
    result = np.empty((size, 3), dtype=np.float32)
    initial_palette_size = np.uint16(initial_palette.shape[0])
    result[:initial_palette_size] = initial_palette

    for i in range(initial_palette_size):
        _ = get_next_color(distances, colors, result[i])

    for i in range(initial_palette_size, size):
        result[i] = get_next_color(distances, colors, result[i - 1])

    return result


@numba.njit(
    [
        #         'f4[::1](f4[:,::1], f4[::1], f4[:,::1])',
        numba.types.Array(numba.float32, 1, "C", readonly=True)(
            numba.types.Array(numba.float32, 2, "C", readonly=True),
            numba.types.Array(numba.float32, 1, "C"),
            numba.types.Array(numba.float32, 2, "C", readonly=True),
        )
    ],
    locals={
        "i": numba.uint16,
    },
)
def generate_next_color_cam02ucs(colors, current_distances, new_palette_block):
    for i in range(new_palette_block.shape[0]):
        _ = get_next_color(current_distances, colors, new_palette_block[i])

    result = get_next_color(current_distances, colors, new_palette_block[0])

    return result
