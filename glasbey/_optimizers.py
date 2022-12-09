import numpy as np


def optimize_endpoint_color(
    color, other_colors, cam_grid, cam_grid_index, compression_direction, scale=1.0, n_iter=30
):
    distances = np.sqrt(np.sum((other_colors - color) ** 2, axis=1))
    closest_idx = np.argmin(distances)
    closest_color = other_colors[closest_idx]
    for val in np.linspace(scale, 0, n_iter):
        if distances[closest_idx] >= 15.0:
            break

        if distances[closest_idx] == 0:
            direction = compression_direction
        else:
            direction = color - closest_color

        direction /= np.sqrt(np.sum(direction**2))
        color += val * (direction)
        color = cam_grid[
            np.squeeze(cam_grid_index.kneighbors([color], return_distance=False))
        ]
        distances = np.sqrt(np.sum((other_colors - color) ** 2, axis=1))
        closest_idx = np.argmin(distances)
        closest_color = other_colors[closest_idx]

    return color
