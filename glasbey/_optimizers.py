import numpy as np
import numba
import colorspacious
from sklearn.metrics import pairwise_distances

BLACK = colorspacious.cspace_convert([0, 0, 0], "sRGB1", "CAM02-UCS")
WHITE = colorspacious.cspace_convert([1, 1, 1], "sRGB1", "CAM02-UCS")


def optimize_endpoint_color(
    color,
    other_colors,
    cam_grid,
    cam_grid_index,
    compression_direction,
    scale=1.0,
    n_iter=30,
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


@numba.njit()
def closest_pair_in_palette(cam_palette, movable_colors=None):
    """
    Find the closest pair, and the distance they are apart in
    the given palette. If movable_colors is provided, ensures that
    at least one color in the pair is movable.
    """
    min_dist = np.inf
    closest_pair = (-1, -1)

    for i in range(cam_palette.shape[0]):
        for j in range(i + 1, cam_palette.shape[0]):
            # Skip if neither color is movable when movable_colors is specified
            if (
                movable_colors is not None
                and i not in movable_colors
                and j not in movable_colors
            ):
                continue

            color1 = cam_palette[i]
            color2 = cam_palette[j]

            # Calculate the distance between the two colors
            dist = np.sqrt(np.sum((color1 - color2) ** 2))

            # If the distance is less than the current minimum, update the palette
            if dist < min_dist:
                min_dist = dist
                closest_pair = (i, j)

    return closest_pair, min_dist


def find_best_replacement(
    color_index,
    current_palette,
    cam_grid,
    cam_grid_index,
    search_radius=10,
    avoid_black_white=True,
):
    color = current_palette[color_index]
    remaining_colors = np.delete(current_palette, color_index, axis=0)
    if avoid_black_white:
        remaining_colors = np.vstack([remaining_colors, BLACK, WHITE])

    neighbors = cam_grid[
        np.squeeze(
            cam_grid_index.radius_neighbors(
                [color], radius=search_radius, return_distance=False
            )[0]
        )
    ]
    min_distances = pairwise_distances(neighbors, remaining_colors).min(axis=1)
    best_neighbor_idx = np.argmax(min_distances)
    return (min_distances[best_neighbor_idx], neighbors[best_neighbor_idx], color_index)


def optimize_existing_palette(
    cam_palette,
    cam_grid,
    cam_grid_index,
    movable_colors=None,
    avoid_black_white=True,
    search_radius=10,
    n_iter=100,
):
    """
    Optimize the palette by moving colors towards their nearest neighbors in the grid.
    This is a greedy algorithm that may not find the global optimum, but it should
    be fast and effective for small palettes.

    Parameters
    ----------
    cam_palette : np.ndarray
        The palette to optimize in CAM02-UCS space, shape (n_colors, 3).

    cam_grid : np.ndarray
        The CAM02-UCS grid to use for nearest neighbor searches, shape (n_grid_points, 3).

    cam_grid_index : NearestNeighbors
        The nearest neighbors index for the CAM02-UCS grid.

    movable_colors : list or None
        Indices of palette items that can be changed. If None, all colors can be changed.

    n_iter : int
        Maximum number of iterations for optimization.
    """
    current_palette = cam_palette.copy()
    current_swap_candidates, current_min_dist = closest_pair_in_palette(
        current_palette, movable_colors
    )

    for n in range(n_iter):
        # Find best replacement for each color in the closest pair
        i, j = current_swap_candidates

        # Only consider movable colors
        if movable_colors is not None:
            can_move_i = i in movable_colors
            can_move_j = j in movable_colors

            if not (can_move_i or can_move_j):
                # Neither color can be moved, so we're done
                break

            if can_move_i and not can_move_j:
                # Only i can be moved
                candidate_min_dist, candidate_swap_color, candidate_index = (
                    find_best_replacement(
                        i,
                        current_palette,
                        cam_grid,
                        cam_grid_index,
                        avoid_black_white=avoid_black_white,
                        search_radius=search_radius,
                    )
                )
            elif can_move_j and not can_move_i:
                # Only j can be moved
                candidate_min_dist, candidate_swap_color, candidate_index = (
                    find_best_replacement(
                        j,
                        current_palette,
                        cam_grid,
                        cam_grid_index,
                        avoid_black_white=avoid_black_white,
                        search_radius=search_radius,
                    )
                )
            else:
                # Both can be moved, choose the better one
                candidate_min_dist_0, candidate_swap_color_0, _ = find_best_replacement(
                    i,
                    current_palette,
                    cam_grid,
                    cam_grid_index,
                    avoid_black_white=avoid_black_white,
                    search_radius=search_radius,
                )
                candidate_min_dist_1, candidate_swap_color_1, _ = find_best_replacement(
                    j,
                    current_palette,
                    cam_grid,
                    cam_grid_index,
                    avoid_black_white=avoid_black_white,
                    search_radius=search_radius,
                )

                # Choose the better replacement
                if candidate_min_dist_0 > candidate_min_dist_1:
                    candidate_min_dist = candidate_min_dist_0
                    candidate_swap_color = candidate_swap_color_0
                    candidate_index = i
                else:
                    candidate_min_dist = candidate_min_dist_1
                    candidate_swap_color = candidate_swap_color_1
                    candidate_index = j
        else:
            # All colors can be moved, original behavior
            candidate_min_dist_0, candidate_swap_color_0, _ = find_best_replacement(
                i,
                current_palette,
                cam_grid,
                cam_grid_index,
                avoid_black_white=avoid_black_white,
                search_radius=search_radius,
            )
            candidate_min_dist_1, candidate_swap_color_1, _ = find_best_replacement(
                j,
                current_palette,
                cam_grid,
                cam_grid_index,
                avoid_black_white=avoid_black_white,
                search_radius=search_radius,
            )

            # Choose the better replacement
            if candidate_min_dist_0 > candidate_min_dist_1:
                candidate_min_dist = candidate_min_dist_0
                candidate_swap_color = candidate_swap_color_0
                candidate_index = i
            else:
                candidate_min_dist = candidate_min_dist_1
                candidate_swap_color = candidate_swap_color_1
                candidate_index = j

        # Only perform swap if it improves minimum distance
        if (
            candidate_min_dist > current_min_dist
            and np.abs(candidate_min_dist - current_min_dist) > 1e-8
        ):
            current_palette[candidate_index] = candidate_swap_color
            current_swap_candidates, current_min_dist = closest_pair_in_palette(
                current_palette, movable_colors
            )
        else:
            # If no swap was made, we can stop the optimization
            break

    return current_palette
