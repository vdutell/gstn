# ---------------------------------------------------------------------------
# Produces: contrastive compressor training
#
# Provenance: StatTexNetExtended/train_compressor.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""
Train the StatTexGramNet contrastive compressor at a parametric bottleneck
size. Mirrors StaTexGramNet.ipynb's training loop exactly, with hardcoded
paths replaced for this machine.

Env-vars:
    BOTTLENECK_SIZE   bottleneck dim (default 100)
    NUM_EPOCHS        default 100
    GPU               cuda:0 / cuda:1 (default cuda:0)
    OUT_ROOT          where to write the saved checkpoints

Output:
    saved_outputs/run_btch45_crp128_size_bns{N}_lr0.01_normTrue/final_model_checkpoint.pth
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import DTD, REPO, require  # noqa: E402

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import utils.stattexgramnet as stn

DEVICE = torch.device(os.environ.get("GPU", "cuda:0"))
BOTTLENECK_SIZE = int(os.environ.get("BOTTLENECK_SIZE", "100"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "100"))
OUT_ROOT = Path(os.environ.get("OUT_ROOT",
                               REPO / "saved_outputs"))

# Hyperparams (match StaTexGramNet.ipynb)
LEARNING_RATE = 1e-2
BATCH_SIZE = 45
CROP_SIZE = 128
NUM_CROPS = 5
COLOR = True
ORIS = 4
EDGE_LEVELS = 6
EDGE_START = 1
ADD_MARGINALS = True
INCLUDE_DPHASE = False
BOUNDARY_MODE = "wrap"
NORM = True

# torchvision.datasets.DTD expects a nested layout, <root>/dtd/dtd/{images,labels},
# which is not how DTD ships. Rather than require the user to hand-build that tree
# (the original working copy relied on a /tmp symlink that did not survive reboots),
# materialize it once as a symlink under build/.
def _torchvision_dtd_root() -> str:
    real = require(DTD, "DTD dataset root")
    root = REPO / "build" / "dtd_torchvision"
    nested = root / "dtd" / "dtd"
    nested.parent.mkdir(parents=True, exist_ok=True)
    if nested.is_symlink() or nested.exists():
        if nested.resolve() == real.resolve():
            return str(root)
        nested.unlink()
    nested.symlink_to(real.resolve(), target_is_directory=True)
    return str(root)


DTD_ROOT = _torchvision_dtd_root()

torch.manual_seed(5)


def main():
    print(f"== Training compressor: BOTTLENECK_SIZE={BOTTLENECK_SIZE}, "
          f"NUM_EPOCHS={NUM_EPOCHS}, GPU={DEVICE}")

    save_dir = OUT_ROOT / (f"run_flipflip_btch{BATCH_SIZE}_crp{CROP_SIZE}"
                           f"_size_bns{BOTTLENECK_SIZE}_lr{LEARNING_RATE}_norm{NORM}")
    save_dir.mkdir(parents=True, exist_ok=True)

    loading_transforms = transforms.Compose([
        transforms.FiveCrop(size=CROP_SIZE),
        transforms.Lambda(lambda crops: torch.stack(
            [transforms.PILToTensor()(c) for c in crops])),
        transforms.ConvertImageDtype(torch.float32),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    dtd_dataset = torchvision.datasets.DTD(
        root=DTD_ROOT, split="train", partition=1,
        transform=loading_transforms, target_transform=None,
        download=False,
    )
    sampler = data.RandomSampler(dtd_dataset)
    dtd_loader = DataLoader(dtd_dataset, sampler=sampler,
                            batch_size=BATCH_SIZE, shuffle=False)

    model = stn.StatTexGramNet(
        img_size=(CROP_SIZE, CROP_SIZE),
        batch_size=BATCH_SIZE,
        bottleneck_size=BOTTLENECK_SIZE,
        device=DEVICE,
        compress=True,
        color=COLOR,
        add_marginals=ADD_MARGINALS,
        oris=ORIS,
        edge_levels=EDGE_LEVELS,
        edge_start=EDGE_START,
        include_dphase=INCLUDE_DPHASE,
        boundary_mode=BOUNDARY_MODE,
    ).to(DEVICE)

    # Sanity check: stats_size matches the actual forward output (color marginal
    # bug fixed in stattexgramnet.py).
    with torch.no_grad():
        priming_input = torch.zeros(1, 3, CROP_SIZE, CROP_SIZE, device=DEVICE)
        encoder_out = model.encoder(priming_input)
    assert encoder_out.shape[1] == model._w.shape[0], (
        f"stats_size mismatch: forward={encoder_out.shape[1]} vs "
        f"_w={model._w.shape[0]}")

    loss_fn = losses.NTXentLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Prime: needed because the model has stateful sub-buffers
    for batch in dtd_loader:
        x = torch.flatten(batch[0].to(DEVICE), start_dim=0, end_dim=1)
        _ = model.forward(x)
        break

    training_loss = []
    model.train()

    print(f"Saving to {save_dir}")
    for epoch in range(NUM_EPOCHS):
        for j, batch in enumerate(dtd_loader):
            x = torch.flatten(batch[0].to(DEVICE), start_dim=0, end_dim=1)
            actual_bs = batch[0].shape[0]
            labels = torch.repeat_interleave(torch.arange(actual_bs), NUM_CROPS)
            perm = torch.randperm(actual_bs * NUM_CROPS)
            x = x[perm]
            labels = labels[perm].to(DEVICE)

            optimizer.zero_grad()
            embedding = model.forward(x)
            loss = loss_fn(embedding, labels)
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
            if j % 30 == 0:
                print(f"  epoch {epoch:3d} batch {j:3d}  loss={loss.item():.4f}",
                      flush=True)

        # Periodic checkpoint
        if epoch % 20 == 0 and epoch > 0:
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, save_dir / f"model_checkpoint_epoch_{epoch}.pth")

    torch.save({
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, save_dir / "final_model_checkpoint.pth")
    import json
    (save_dir / "training_loss.json").write_text(
        json.dumps({"loss": training_loss}, indent=2))
    print(f"== done. Final ckpt: {save_dir / 'final_model_checkpoint.pth'}")


if __name__ == "__main__":
    main()
