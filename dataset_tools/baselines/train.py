"""
Reference training loop tying dataset.py + fusion_models.py together, for
all three released baseline variants: RGB-only, IR-only, and mid-fusion.
(Early fusion trains identically to RGB-only/IR-only but with a
`SingleModalityDetector(in_ch=4)` on `fusion_mode="early"` data -- see the
CLI's `--variant early` handling below. Late fusion doesn't need a training
script: train `rgb` and `ir` separately, then combine at inference with
late_fusion_wbf.py.)

Target assignment is intentionally simple (center-cell assignment, single
scale, no anchors) to keep this auditable as a *reference* baseline rather
than a competitive one -- see the module docstring in fusion_models.py for
why that's the right scope for this release. Swap in Ultralytics YOLO (via
patch_yolo_early_fusion for the early-fusion variant) for a competitive
number.

NOT executed in this sandbox (needs torch + the real corpus) -- see
PLAN.md / docs/RELEASE_RUNBOOK.md. Sanity-check before a real run with:

    python -m dataset_tools.baselines.train --self-test
"""

import argparse

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ImportError:
    torch = None

from .dataset import AVIDPairDataset
from .fusion_models import MidFusionDetector, SingleModalityDetector


def build_targets(boxes, labels, grid_h, grid_w, n_classes):
    """boxes: (N, 4) cx,cy,w,h in [0,1]. Returns a (5+n_classes, grid_h,
    grid_w) target tensor: channel 0 = objectness, 1:5 = box (tx,ty,tw,th)
    relative to the assigned cell, 5: = one-hot class, assigned to the
    single grid cell containing each box's center (cells with >1 box keep
    the last one -- acceptable for this dataset's largely single/sparse
    frames; a real training run should log how often that collision fires)."""
    target = torch.zeros(5 + n_classes, grid_h, grid_w)
    for (cx, cy, w, h), label in zip(boxes.tolist(), labels.tolist()):
        gx, gy = int(cx * grid_w), int(cy * grid_h)
        gx, gy = min(gx, grid_w - 1), min(gy, grid_h - 1)
        target[0, gy, gx] = 1.0
        target[1, gy, gx] = cx * grid_w - gx
        target[2, gy, gx] = cy * grid_h - gy
        target[3, gy, gx] = w
        target[4, gy, gx] = h
        target[5 + label, gy, gx] = 1.0
    return target


def detection_loss(pred, targets, n_classes):
    """pred: (B, 5+n_classes, H, W) raw logits. targets: list of per-sample
    (boxes, labels) from the dataset's collate step, already turned into a
    stacked (B, 5+n_classes, H, W) tensor by the caller."""
    obj_logit = pred[:, 0]
    box_pred = pred[:, 1:5]
    cls_logit = pred[:, 5:]

    obj_target = targets[:, 0]
    box_target = targets[:, 1:5]
    cls_target = targets[:, 5:]

    obj_loss = F.binary_cross_entropy_with_logits(obj_logit, obj_target)
    mask = obj_target.unsqueeze(1)
    box_loss = (F.smooth_l1_loss(box_pred, box_target, reduction="none") * mask).sum() / (mask.sum() * 4 + 1e-6)
    cls_loss = (
        F.binary_cross_entropy_with_logits(cls_logit, cls_target, reduction="none") * obj_target.unsqueeze(1)
    ).sum() / (obj_target.sum() * n_classes + 1e-6)
    return obj_loss + 5.0 * box_loss + cls_loss


def collate_fn(batch, grid_hw, n_classes):
    samples, meta_targets = zip(*batch)
    if isinstance(samples[0], tuple):
        rgb_batch = torch.stack([s[0] for s in samples])
        ir_batch = torch.stack([s[1] for s in samples])
        inputs = (rgb_batch, ir_batch)
    else:
        inputs = torch.stack(samples)
    targets = torch.stack(
        [build_targets(t["boxes"], t["labels"], grid_hw[0], grid_hw[1], n_classes) for t in meta_targets]
    )
    return inputs, targets


def train(args):
    if torch is None:
        raise ImportError("training requires PyTorch: pip install torch")

    n_classes = 2  # drone, bird
    grid_hw = (args.img_h // 32, args.img_w // 32)

    if args.variant == "rgb":
        model = SingleModalityDetector(in_ch=3, n_classes=n_classes)
        fusion_mode = "rgb"
    elif args.variant == "ir":
        model = SingleModalityDetector(in_ch=1, n_classes=n_classes)
        fusion_mode = "ir"
    elif args.variant == "early":
        model = SingleModalityDetector(in_ch=4, n_classes=n_classes)
        fusion_mode = "early"
    elif args.variant == "mid_fusion":
        model = MidFusionDetector(n_classes=n_classes)
        fusion_mode = "dual"
    else:
        raise ValueError(args.variant)

    train_ds = AVIDPairDataset(args.root, "train", fusion_mode=fusion_mode, img_size=(args.img_h, args.img_w))
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=lambda b: collate_fn(b, grid_hw, n_classes),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for step, (inputs, targets) in enumerate(loader):
            targets = targets.to(device)
            if isinstance(inputs, tuple):
                pred = model(inputs[0].to(device), inputs[1].to(device))
            else:
                pred = model(inputs.to(device))
            loss = detection_loss(pred, targets, n_classes)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
        print(f"epoch {epoch} mean loss {running / max(1, len(loader)):.4f}")
        torch.save(model.state_dict(), f"{args.variant}_epoch{epoch}.pt")


def _self_test():
    """No dataset needed: builds a tiny synthetic batch matching the same
    tensor/target shapes `train()` produces, and confirms one optimization
    step reduces the loss -- catching shape/wiring bugs before a real run."""
    assert torch is not None, "install torch to run this self-test: pip install torch"
    n_classes, grid_hw = 2, (8, 10)
    boxes = torch.tensor([[0.5, 0.5, 0.05, 0.05]])
    labels = torch.tensor([0])
    target = build_targets(boxes, labels, *grid_hw, n_classes).unsqueeze(0)

    model = SingleModalityDetector(in_ch=3, n_classes=n_classes)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    x = torch.randn(1, 3, 256, 320)

    losses = []
    for _ in range(20):
        pred = model(x)
        loss = detection_loss(pred, target, n_classes)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    print(f"train.py self-test passed: loss {losses[0]:.4f} -> {losses[-1]:.4f} over 20 steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="dataset root (see hf_release/SCHEMA.md)")
    parser.add_argument("--variant", choices=["rgb", "ir", "early", "mid_fusion"], default="mid_fusion")
    parser.add_argument("--img-h", type=int, default=256)
    parser.add_argument("--img-w", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
    else:
        if not args.root:
            parser.error("--root is required unless --self-test is passed")
        train(args)
