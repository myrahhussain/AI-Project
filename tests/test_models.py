import numpy as np
import pytest
import torch
import torchvision

from src import models, train


class FakeCallback:
    """Stands in for a PyTorch Lightning object saved inside a checkpoint."""

    def __init__(self):
        self.best = 1.0


@pytest.mark.parametrize("backbone", models.BACKBONES)
def test_every_backbone_gives_one_logit_per_patch(backbone):
    m = models.make_model(pretrained=False, backbone=backbone).eval()
    with torch.no_grad():
        out = m(torch.rand(2, 3, 96, 96))
    assert out.shape == (2,)


def test_resnet18_layout_is_unchanged_so_interim_checkpoints_load():
    keys = set(models.make_model(pretrained=False).state_dict())
    tv = torchvision.models.resnet18(weights=None)
    tv.fc = torch.nn.Linear(512, 1)
    assert keys == {"mean", "std"} | {"backbone." + k for k in tv.state_dict()}


def test_unknown_backbone_is_rejected():
    with pytest.raises(ValueError):
        models.make_model(pretrained=False, backbone="vgg16")


def _lightning_style_checkpoint(path, net):
    """Save weights the way the Ciga et al. release does: prefixed keys, an extra head, other objects."""
    sd = {"model.resnet." + k: v for k, v in net.state_dict().items() if not k.startswith("fc.")}
    sd["model.resnet.fc.1.weight"] = torch.zeros(4, 512)
    torch.save({"epoch": 1000, "callbacks": {"ckpt": FakeCallback()}, "state_dict": sd}, path)


def test_histology_resnet18_weights_load_without_running_pickled_code(tmp_path):
    src = torchvision.models.resnet18(weights=None)
    path = tmp_path / "tenpercent_resnet18.ckpt"
    _lightning_style_checkpoint(path, src)
    state = models.load_state_dict_file(str(path))
    assert all(torch.is_tensor(v) for v in state.values())
    m = models.make_model(pretrained=True, backbone="resnet18_histo", weights_path=str(path))
    assert torch.equal(m.backbone.conv1.weight, src.conv1.weight)
    assert torch.equal(m.backbone.layer4[1].bn2.running_var, src.layer4[1].bn2.running_var)
    assert m.backbone.fc.out_features == 1


def test_histology_resnet50_plain_state_dict_loads(tmp_path):
    src = torchvision.models.resnet50(weights=None)
    path = tmp_path / "mocov2_rn50_ep200.torch"
    torch.save({k: v for k, v in src.state_dict().items() if not k.startswith("fc.")}, path)
    m = models.make_model(pretrained=True, backbone="resnet50_histo", weights_path=str(path))
    assert torch.equal(m.backbone.layer3[0].conv2.weight, src.layer3[0].conv2.weight)


def test_wrong_weight_file_fails_loudly(tmp_path):
    path = tmp_path / "wrong.torch"
    torch.save(torchvision.models.resnet50(weights=None).state_dict(), path)
    with pytest.raises(RuntimeError):
        models.make_model(pretrained=True, backbone="resnet18_histo", weights_path=str(path))


def test_histology_backbone_without_weights_path_is_rejected():
    with pytest.raises(ValueError):
        models.make_model(pretrained=True, backbone="resnet50_histo")


def test_model_from_run_restores_the_same_predictions(tmp_path):
    m = models.make_model(pretrained=False, backbone="resnet18_histo").eval()
    ck = tmp_path / "run.pt"
    torch.save(m.state_dict(), ck)
    again = train.model_from_run({"backbone": "resnet18_histo"}, str(ck), torch.device("cpu"))
    x = np.random.default_rng(0).integers(0, 255, (4, 96, 96, 3), dtype=np.uint8)
    p1 = train.predict(m, x, torch.device("cpu"))
    p2 = train.predict(again, x, torch.device("cpu"))
    assert np.allclose(p1, p2, atol=1e-6)
