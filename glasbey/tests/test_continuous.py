import numpy as np
import pytest
from colorspacious import cspace_convert
from matplotlib.colors import to_rgb
from scipy.interpolate import make_interp_spline

from glasbey._continuous import _Arc, _Domain, _Grid, _fit_arm, _arm_diagnostics, _cam02ucs_to_jch
from glasbey import create_colormap


def test_two_seed_arc():
    seeds = np.array([[0.2, 0.25, 0.3], [0.75, 0.8, 0.85]])
    anchors = cspace_convert(seeds, "sRGB1", "CAM02-UCS")
    arc = _Arc(make_interp_spline([0, 1], anchors, k=3, bc_type="natural"), [0, 1])
    colors = arc(np.linspace(0, 1, 256))
    np.testing.assert_allclose(colors[[0, -1]], anchors, atol=1e-10)
    distances = np.linalg.norm(np.diff(colors, axis=0), axis=1)
    assert distances.std() / distances.mean() < 1e-6
    assert np.all(np.diff(colors[:, 0]) >= 0)
    assert _Domain().contains(colors).all()


def test_arc_preserves_interior_anchor():
    anchors = cspace_convert(
        [[0.2, 0.25, 0.3], [0.4, 0.5, 0.6], [0.75, 0.8, 0.85]],
        "sRGB1", "CAM02-UCS",
    )
    parameters = [0, 0.37, 1]
    arc = _Arc(make_interp_spline(parameters, anchors, bc_type="natural"), parameters)
    np.testing.assert_allclose(arc(arc.anchor_positions), anchors, atol=1e-10)


def test_domain_does_not_clip():
    colors = cspace_convert([[1.1, 0.5, 0.5]], "sRGB1", "CAM02-UCS")
    assert not _Domain().contains(colors).any()


def test_analytic_jch_matches_colorspacious():
    rgb = np.random.default_rng(0).random((500, 3))
    expected = cspace_convert(rgb, "sRGB1", "JCh")
    actual = _cam02ucs_to_jch(cspace_convert(rgb, "sRGB1", "CAM02-UCS"))
    np.testing.assert_allclose(actual[:, :2], expected[:, :2], atol=1e-9)
    hue_error = np.abs(actual[:, 2] - expected[:, 2])
    assert np.all(np.minimum(hue_error, 360 - hue_error) < 1e-9)


def test_fitted_arm_preserves_anchors_and_gamut():
    anchors = cspace_convert(
        [[0.2, 0.25, 0.3], [0.4, 0.5, 0.6], [0.75, 0.8, 0.85]],
        "sRGB1", "CAM02-UCS",
    )
    domain = _Domain()
    fitted = _fit_arm(anchors, _Grid(domain, 12, "RGB", anchors), 8, 1.3, 1, 0, 100)
    arc, diagnostics = fitted.arc, _arm_diagnostics(fitted)
    colors = arc(np.linspace(0, 1, 513))
    np.testing.assert_allclose(arc(arc.anchor_positions), anchors, atol=1e-7)
    assert domain.contains(colors).all()
    assert np.min(np.diff(colors[:, 0])) >= -1e-6
    assert diagnostics["length_ratio"] <= 1.3 + 1e-6
    assert diagnostics["step_distance_cv"] < 0.01


@pytest.fixture(scope="module")
def sequential():
    return create_colormap(
        ["#bdd8e8", "#324555", "#64809b"], grid_size=12, max_iter=80,
    )


def test_public_sequential_sampling(sequential):
    positions, rgb = sequential.sample()
    assert rgb.shape == (256, 3)
    np.testing.assert_allclose(positions, np.linspace(0, 1, 256))
    np.testing.assert_allclose(
        sequential(sequential.seed_positions),
        [[189 / 255, 216 / 255, 232 / 255], [50 / 255, 69 / 255, 85 / 255],
         [100 / 255, 128 / 255, 155 / 255]], atol=1e-6,
    )
    cam = cspace_convert(rgb, "sRGB1", "CAM02-UCS")
    assert (np.diff(cam[:, 0]) >= -1e-6).all()
    distances = np.linalg.norm(np.diff(cam, axis=0), axis=1)
    assert distances.std() / distances.mean() < 0.01
    positions, colors = sequential.sample(8, include_seeds=True)
    assert set(sequential.seed_positions) <= set(positions)
    assert len(colors) >= 8
    np.testing.assert_allclose(sequential.to_matplotlib(n_colors=32).colors, sequential.sample(32)[1])
    assert all(color.startswith("#") for color in sequential.sample(8, as_hex=True)[1])


def test_diverging_center_and_seeds():
    seeds = ["#426ba0", "#eee8e0", "#a35742"]
    cmap = create_colormap(seeds, mode="diverging", grid_size=12, max_iter=80)
    assert cmap.seed_positions[1] == 0.5
    np.testing.assert_allclose(cmap(cmap.seed_positions), [to_rgb(color) for color in seeds], atol=1e-6)
    cam = cspace_convert(cmap(np.linspace(0, 1, 513)), "sRGB1", "CAM02-UCS")
    assert np.all(np.diff(cam[:257, 0]) >= -1e-6)
    assert np.all(np.diff(cam[256:, 0]) <= 1e-6)
    assert len(cmap.diagnostics["arms"]) == 2


@pytest.mark.parametrize("options", [
    {"mode": "unknown"}, {"grid_size": 1}, {"grid_size": (4, 4)},
    {"chroma_weight": -1}, {"smoothness": 0}, {"arc_length_factor": 0.5},
    {"red_bounds": (-1, 1)}, {"max_control_points": 3},
])
def test_invalid_parameters(options):
    with pytest.raises(ValueError):
        create_colormap(["#333333", "#dddddd"], **options)


def test_seed_bounds_are_not_relaxed():
    with pytest.raises(ValueError, match="seed colors must satisfy"):
        create_colormap(["#333333", "#dddddd"], lightness_bounds=(40, 60))


def test_chroma_reward_increases_vibrancy():
    settings = dict(grid_size=12, arc_length_factor=1.3, smoothness=3, max_iter=80)
    muted = create_colormap(["#333333", "#dddddd"], chroma_weight=0, **settings)
    vibrant = create_colormap(["#333333", "#dddddd"], chroma_weight=5, **settings)
    assert vibrant.diagnostics["arms"][0]["weighted_chroma"] > muted.diagnostics["arms"][0]["weighted_chroma"] + 1
    assert vibrant.diagnostics["arms"][0]["length_ratio"] <= 1.3 + 1e-6
    np.testing.assert_allclose(vibrant([0, 1]), muted([0, 1]), atol=1e-6)
    assert vibrant.diagnostics["arms"][0]["step_distance_cv"] < 0.01


@pytest.mark.parametrize("grid_space", ["RGB", "JCh"])
def test_wrapped_hue_continuous_constraints(grid_space):
    seeds = cspace_convert([[35, 20, 355], [70, 25, 10]], "JCh", "sRGB1")
    cmap = create_colormap(
        seeds, grid_size=(12, 16, 20), grid_space=grid_space,
        hue_bounds=(350, 20), chroma_bounds=(10, 35), lightness_bounds=(30, 75),
        chroma_weight=2, max_iter=80,
    )
    positions = np.linspace(0, 1, 1001)
    jch = cspace_convert(cmap(positions), "sRGB1", "JCh")
    assert np.all((jch[:, 2] >= 350 - 1e-3) | (jch[:, 2] <= 20 + 1e-3))
    assert jch[:, 1].min() >= 10 - 1e-5
    assert jch[:, 1].max() <= 35 + 1e-5


def test_permutation_duplicates_and_direction():
    settings = dict(grid_size=8, chroma_weight=0, arc_length_factor=1.05, max_iter=60)
    seeds = ["#445566", "#8899aa", "#ddeeff"]
    forward = create_colormap(seeds, **settings)
    permuted = create_colormap([seeds[2], seeds[0], seeds[1], seeds[0]], **settings)
    backward = create_colormap(seeds, direction="decreasing", **settings)
    np.testing.assert_allclose(forward(np.linspace(0, 1, 17)), permuted(np.linspace(0, 1, 17)), atol=1e-7)
    assert permuted.seed_positions[1] == permuted.seed_positions[3]
    backward_cam = cspace_convert(backward(np.linspace(0, 1, 101)), "sRGB1", "CAM02-UCS")
    assert np.all(np.diff(backward_cam[:, 0]) <= 1e-6)


def test_dark_diverging_and_invalid_center():
    seeds = ["#202020", "#7196b8", "#cb9f87"]
    cmap = create_colormap(
        seeds, mode="diverging", center_lightness="dark", center_color=seeds[0],
        grid_size=8, chroma_weight=0, max_iter=60,
    )
    assert cmap.seed_positions[0] == 0.5
    np.testing.assert_allclose(cmap(cmap.seed_positions), [to_rgb(color) for color in seeds], atol=1e-6)
    with pytest.raises(ValueError, match="extreme lightness"):
        create_colormap(seeds, mode="diverging", center_color=seeds[0])


@pytest.mark.parametrize("seeds", [[], ["#333333"], ["#333333", "#333333"], [[np.nan, 0, 0]], [[1, 2]], [object()]])
def test_invalid_seeds(seeds):
    with pytest.raises(ValueError):
        create_colormap(seeds)


@pytest.mark.parametrize("positions", [-0.1, 1.1, np.nan, [0.2, np.inf]])
def test_invalid_evaluation(sequential, positions):
    with pytest.raises(ValueError):
        sequential(positions)


@pytest.mark.parametrize("grid_space,grid_size", [("JCh", (16, 8, 32)), ("RGB", 32)])
def test_route_around_chroma_hole(grid_space, grid_size):
    anchors = cspace_convert([[35, 30, 350], [70, 30, 100]], "JCh", "CAM02-UCS")
    domain = _Domain(chroma_bounds=(25, 40), hue_bounds=(340, 120))
    fraction = np.linspace(0, 1, 65)[:, None]
    assert not domain.contains(anchors[0] + fraction * (anchors[1] - anchors[0])).all()
    path = _Grid(domain, grid_size, grid_space, anchors).connect(*anchors)
    assert np.all(np.diff(path[:, 0]) >= -1e-9)
    for start, end in zip(path[:-1], path[1:]):
        assert domain.contains(start + fraction * (end - start)).all()


def test_equal_lightness_seeds():
    seeds = cspace_convert([[55, 20, 20], [55, 20, 80]], "JCh", "sRGB1")
    cmap = create_colormap(seeds, grid_size=8, chroma_weight=0, max_iter=50)
    np.testing.assert_allclose(cmap(cmap.seed_positions), seeds, atol=1e-6)
    lightness = cspace_convert(cmap(np.linspace(0, 1, 101)), "sRGB1", "CAM02-UCS")[:, 0]
    assert np.ptp(lightness) < 1e-5