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


def test_fallbacks_do_not_crash():
    ref = render(make_concentrations(seed=9), W_C)
    norm = N.StainNormalizer().fit(ref)
    white = np.full((96, 96, 3), 255, np.uint8)
    assert np.array_equal(norm.transform(white), white)
    tiny = white.copy()
    tiny[:3, :3] = [120, 60, 150]
    out = norm.transform(tiny)
    assert out.shape == tiny.shape and out.dtype == np.uint8
    flat = np.full((96, 96, 3), 128, np.uint8)  # grey: not a stain mixture
    assert norm.transform(flat).shape == flat.shape


def test_slide_mode_reduces_slide_to_slide_colour_difference():
    ref = render(make_concentrations(seed=9), W_C)
    norm = N.StainNormalizer().fit(ref)
    patches, slides = [], []
    for s, w in enumerate((W_A, W_B)):
        for k in range(6):
            patches.append(render(make_concentrations(seed=20 + k), w))
            slides.append(s)
    x, slides = np.stack(patches), np.array(slides)
    out = norm.transform_batch(x, slides, mode="slide")
    mean_color = lambda arr, s: arr[slides == s].reshape(-1, 3).mean(0)
    before = np.abs(mean_color(x, 0) - mean_color(x, 1)).sum()
    after = np.abs(mean_color(out, 0) - mean_color(out, 1)).sum()
    assert after < 0.4 * before


def test_patch_mode_matches_batch_shape_and_is_deterministic():
    ref = render(make_concentrations(seed=9), W_C)
    norm = N.StainNormalizer().fit(ref)
    x = np.stack([render(make_concentrations(seed=s), W_B) for s in range(3)])
    a = norm.transform_batch(x, mode="patch")
    b = norm.transform_batch(x, mode="patch")
    assert a.shape == x.shape and np.array_equal(a, b)


def test_choose_reference_returns_valid_index_with_tissue():
    x = np.stack([render(make_concentrations(seed=s), W_A) for s in range(10)])
    i = N.choose_reference(x, n_candidates=10)
    assert 0 <= i < 10
    assert N.tissue_mask(x[i]).mean() > 0.6


def test_estimate_is_robust_to_pixel_noise():
    rng = np.random.default_rng(0)
    conc = make_concentrations(seed=4)
    for w_true in (W_A, W_B):
        img = render(conc, w_true).astype(float) + rng.normal(0, 3, (96, 96, 3))
        img = np.clip(img, 0, 255).astype(np.uint8)
        w, ok = N.estimate_stain_matrix(N.rgb_to_od(img[N.tissue_mask(img)]))
        assert ok
        assert cos(w[0], w_true[0]) > 0.97 and cos(w[1], w_true[1]) > 0.97


def test_choose_reference_avoids_pale_outliers():
    """Twelve typical patches and three pale ones with weak haematoxylin: the reference must be typical."""
    typical, pale = [], []
    for s in range(12):
        typical.append(render(make_concentrations(seed=s), W_A))
    for s in range(12, 15):
        c = make_concentrations(seed=s)
        c[..., 0] *= 0.3                       # weak haematoxylin, nuclei barely stained
        pale.append(render(c, W_A))
    x = np.stack(typical + pale)
    for seed in range(3):
        assert N.choose_reference(x, n_candidates=15, seed=seed) < 12


def test_saturated_dark_pixels_do_not_wash_out_a_slide():
    """Real slides can have a few percent of near-black pixels (very dark nuclei, debris). Those sit at the
    clipping floor and must not set the slide's stain-strength estimate: the slide would be scaled down
    and come out pale (seen on real Camelyon17 slides)."""
    rng = np.random.default_rng(0)
    clean = np.stack([render(make_concentrations(seed=s), W_A) for s in range(8)])
    dark = clean.copy()
    for img in dark:
        mask = rng.random(img.shape[:2]) < 0.04          # 4% of pixels saturated
        img[mask] = [4, 2, 8]
    ref = N.StainNormalizer().fit(render(make_concentrations(seed=99), W_A))
    slides = np.zeros(len(clean), dtype=int)
    out_clean = ref.transform_batch(clean, slides, mode="slide").astype(float)
    out_dark = ref.transform_batch(dark, slides, mode="slide").astype(float)
    keep = (dark.min(axis=-1) > 10)                       # unsaturated pixels only
    lum = lambda a: a @ np.array([0.299, 0.587, 0.114])
    # nuclei are the darkest tissue: they must stay dark, not wash out
    dark_clean = np.percentile(lum(out_clean)[keep], 5)
    dark_sat = np.percentile(lum(out_dark)[keep], 5)
    assert abs(dark_clean - dark_sat) < 15, (dark_clean, dark_sat)
