import numpy as np
import pandas as pd
import pytest

from src import size

STEP = 32           # WILDS patch spacing


def cells_to_coords(cells, x0=1000, y0=5000, step=STEP):
    cells = np.asarray(cells)
    return x0 + cells[:, 0] * step, y0 + cells[:, 1] * step


def test_grid_step_and_grid_positions():
    x = np.array([1000, 1032, 1128, 1000])
    y = np.array([64, 96, 64, 160])
    assert size.grid_step(x) == 32
    gx, gy = size.to_grid(x, y)
    assert gx.tolist() == [0, 1, 4, 0] and gy.tolist() == [0, 1, 0, 3]


def test_square_lesion_size_and_category():
    block = [(i, j) for i in range(10) for j in range(10)]
    x, y = cells_to_coords(block)
    gx, gy = size.to_grid(x, y)
    st = size.lesion_stats(gx, gy, np.ones(len(block), bool))
    expected_mm = (np.sqrt(9 ** 2 + 9 ** 2) + 1) * size.CELL_UM / 1000
    assert st["largest_cells"] == 100 and st["n_groups"] == 1
    assert st["largest_mm"] == pytest.approx(expected_mm)
    assert st["category"] == "micro"                      # ~0.43 mm


def test_largest_group_wins_and_small_groups_are_counted():
    big = [(i, 0) for i in range(80)]                   # a straight line: hull fallback path
    small = [(100, 50), (101, 50)]
    x, y = cells_to_coords(big + small)
    gx, gy = size.to_grid(x, y)
    st = size.lesion_stats(gx, gy, np.ones(len(big) + 2, bool))
    assert st["n_groups"] == 2 and st["largest_cells"] == 80
    assert st["largest_mm"] == pytest.approx(80 * size.CELL_UM / 1000)
    assert st["category"] == "macro"                      # ~2.5 mm


def test_only_positive_cells_count():
    cells = [(0, 0), (1, 0), (2, 0)]
    x, y = cells_to_coords(cells)
    gx, gy = size.to_grid(x, y)
    st = size.lesion_stats(gx, gy, np.array([True, False, False]))
    assert st["largest_cells"] == 1 and st["category"] == "itc"
    none = size.lesion_stats(gx, gy, np.zeros(3, bool))
    assert none["category"] == "negative" and none["largest_mm"] == 0


@pytest.mark.parametrize("mm,cat", [(0, "negative"), (0.05, "itc"), (0.2, "itc"), (0.21, "micro"),
                                    (2.0, "micro"), (2.01, "macro")])
def test_size_categories_follow_the_clinical_cut_offs(mm, cat):
    assert size.size_category(mm) == cat


def test_stage_labels_keep_slides_and_skip_patients(tmp_path):
    f = tmp_path / "stage_labels.csv"
    f.write_text("patient,stage\npatient_004.zip,pN1\npatient_004_node_4.tif,Macro\npatient_004_node_0.tif,negative\n")
    assert size.load_stage_labels(str(f)) == {"patient_004_node_4.tif": "macro", "patient_004_node_0.tif": "negative"}
    assert size.slide_name("004", 4) == "patient_004_node_4.tif"


def test_per_slide_and_compare_count_over_and_under_staging():
    rows = []
    for patient, node, n in (("010", 1, 100), ("020", 2, 4)):
        side = int(np.sqrt(n))
        for i in range(side):
            for j in range(side):
                rows.append({"patient": patient, "node": node, "x_coord": i * STEP, "y_coord": j * STEP})
    meta = pd.DataFrame(rows)
    official = {"patient_010_node_1.tif": "micro", "patient_020_node_2.tif": "itc"}
    truth = size.per_slide(meta, np.ones(len(meta), bool), official=official)
    assert truth.set_index("slide")["category"].to_dict() == official
    # a model that misses everything on the first slide and over-calls nothing
    pred = size.per_slide(meta, np.r_[np.zeros(100, bool), np.ones(4, bool)])
    c = size.compare(pred, truth)
    assert c["slides"] == 2 and c["under_staged"] == 1 and c["over_staged"] == 0
    assert c["category_agreement"] == 0.5 and c["mean_error_mm"] < 0
