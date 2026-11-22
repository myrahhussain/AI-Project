import numpy as np

from src import normalize as N


def unit(v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


W_A = np.stack([unit([0.65, 0.70, 0.29]), unit([0.07, 0.99, 0.11])])       # standard stains
W_B = np.stack([unit([0.55, 0.62, 0.55]), unit([0.20, 0.93, 0.30])])       # different lab: shifted colours
W_C = np.stack([unit([0.72, 0.62, 0.30]), unit([0.04, 0.97, 0.24])])       # a third colour style


def make_concentrations(size=96, seed=0):
    """Nuclei-like blobs (H) on eosin-stained tissue (E), plus a white background corner."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    c = np.zeros((size, size, 2))
    c[..., 1] = 0.6 + 0.2 * rng.random((size, size))      # eosin everywhere
    for _ in range(12):
        cy, cx, r = rng.integers(8, size - 8, 2).tolist() + [rng.integers(3, 7)]
        nuc = ((yy - cy) ** 2 + (xx - cx) ** 2) < r * r
        c[nuc, 0] = 0.9 + 0.2 * rng.random()
        c[nuc, 1] = 0.05  # nuclei are haematoxylin-dominated (at most one effective stain per pixel)
    c[:20, :20] = 0                                         # background corner
    return c


def render(conc, w):
    od = conc.reshape(-1, 2) @ w
    return np.rint(N.od_to_rgb(od)).astype(np.uint8).reshape(conc.shape[0], conc.shape[1], 3)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def mean_abs(a, b):
    return float(np.abs(a.astype(float) - b.astype(float)).mean())


def test_estimate_recovers_known_stain_colours():
    conc = make_concentrations()
    for w_true in (W_A, W_B, W_C):
        img = render(conc, w_true)
        mask = N.tissue_mask(img)
        w, ok = N.estimate_stain_matrix(N.rgb_to_od(img[mask]))
        assert ok
        assert cos(w[0], w_true[0]) > 0.98 and cos(w[1], w_true[1]) > 0.98


def test_stain_order_is_haematoxylin_first():
    img = render(make_concentrations(), W_B)
    w, _ = N.estimate_stain_matrix(N.rgb_to_od(img[N.tissue_mask(img)]))
    assert w[0, 0] / w[0, 1] > w[1, 0] / w[1, 1]


def test_normalization_makes_differently_stained_copies_agree():
    conc = make_concentrations(seed=1)
    a, b = render(conc, W_A), render(conc, W_B)
    ref = render(make_concentrations(seed=9), W_C)
    norm = N.StainNormalizer().fit(ref)
    na, nb = norm.transform(a), norm.transform(b)
    before, after = mean_abs(a, b), mean_abs(na, nb)
    assert before > 5
    assert after < 0.25 * before  # observed ~0.1 across stain-style pairs


def test_normalized_colours_move_towards_reference():
    conc = make_concentrations(seed=2)            # same tissue, different stain colours
    ref, src = render(conc, W_C), render(conc, W_B)
    norm = N.StainNormalizer().fit(ref)
    out = norm.transform(src)
    assert mean_abs(out, ref) < 0.25 * mean_abs(src, ref)  # observed ~0.1


def test_background_pixels_are_unchanged_and_output_is_uint8():
    ref = render(make_concentrations(seed=9), W_C)
    src = render(make_concentrations(seed=2), W_B)
    out = N.StainNormalizer().fit(ref).transform(src)
    assert out.dtype == np.uint8 and out.shape == src.shape
    assert np.array_equal(out[:20, :20], src[:20, :20])
