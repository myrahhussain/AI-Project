import numpy as np
import pytest

from src import data


def synthetic_meta(seed=0, slides_per_hospital=10, patches_per_slide=300):
    """Mimic WILDS: hospitals 0,3,4 train; 1 val; 2 test; id_val shares slides with train."""
    rng = np.random.default_rng(seed)
    hosp, slide, y, split = [], [], [], []
    sid = 0
    for h in range(5):
        for _ in range(slides_per_hospital):
            for _ in range(patches_per_slide):
                hosp.append(h)
                slide.append(sid)
                y.append(int(rng.random() < 0.5))
                if h in (0, 3, 4):
                    split.append(1 if rng.random() < 0.1 else 0)  # id_val from the SAME slides
                elif h == 1:
                    split.append(3)
                else:
                    split.append(2)
            sid += 1
    return data.Meta(
        hospital=np.array(hosp), slide=np.array(slide), y=np.array(y), split=np.array(split),
        split_dict={"train": 0, "id_val": 1, "test": 2, "val": 3},
    )


def test_official_id_val_overlap_is_detected():
    meta = synthetic_meta()
    info = data.check_official_id_val_overlap(meta)
    assert info["shared_slides"] > 0  # the official id_val would leak in this mock


def test_official_split_has_no_slide_leakage_and_no_test_hospital():
    meta = synthetic_meta()
    split = data.make_official_split(meta, n_train=2000, n_id_val=400, n_ood_val=400, seed=1)
    assert not data.shared_slides(meta, split["train"], split["id_val"])
    assert not data.shared_slides(meta, split["train"], split["ood_val"])
    assert "test" not in split
    for name, idx in split.items():
        assert not np.isin(meta.hospital[idx], [data.TEST_HOSPITAL]).any(), name
    assert set(meta.hospital[split["ood_val"]]) == {data.VAL_HOSPITAL}
    assert set(meta.hospital[split["train"]]) <= set(data.TRAIN_HOSPITALS)


def test_every_training_hospital_is_represented_in_id_val():
    meta = synthetic_meta()
    split = data.make_official_split(meta, n_train=2000, n_id_val=900, n_ood_val=300, seed=2)
    assert set(meta.hospital[split["id_val"]]) == set(data.TRAIN_HOSPITALS)
