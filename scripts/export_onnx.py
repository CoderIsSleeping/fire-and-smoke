"""Export an FCOS-head checkpoint to a batched ONNX graph, and prove it is right.

This is the step between a trained PyTorch model and a multi-camera GPU
deployment (TensorRT, DeepStream, or a batching service on ONNX Runtime).

Only the FCOS head can do this. The Faster R-CNN head was measured to block it:
the legacy exporter bakes in batch size 1 (batch 4 fails in ONNX Runtime) and
torch.export fails outright, because the RPN chooses a variable number of
proposals and RoIAlign crops features for them before the box head runs. FCOS
puts every learned layer before anything data-dependent.

The exported graph stops at the raw head outputs. Decoding and NMS run
afterwards through `FireSmokeDetector.decode_raw`, or TensorRT's EfficientNMS
plugin in production.

Two checks run on every export, and the script fails if either does:

  1. raw tensors from ONNX Runtime match PyTorch, at several batch sizes
  2. boxes decoded from the ONNX outputs match the PyTorch model's own
     detections -- the check that actually matters, since matching tensors
     are worthless if decoding them goes wrong

    python scripts/export_onnx.py --weights runs/fire_smoke/light/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch

from fire_smoke.glow import GLOW_FEATURE_DIM
from fire_smoke.model import FireSmokeDetector, load_detector

OUTPUT_NAMES = ["cls_logits", "bbox_regression", "bbox_ctrness", "scene_logits"]


class RawGraph(torch.nn.Module):
    def __init__(self, model: FireSmokeDetector) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor, glow: torch.Tensor):
        return self.model.forward_raw(images, glow)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export an FCOS checkpoint to batched ONNX and verify it.")
    p.add_argument("--weights", default=None, help="FCOS checkpoint. Omit to export an untrained model "
                                                   "of the given config (for pipeline testing only).")
    p.add_argument("--output", default=None, help="Output .onnx path (default: next to the weights).")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--verify-batches", default="1,4", help="Batch sizes to verify in ONNX Runtime.")
    p.add_argument("--backbone", default="vit_tiny_patch16_dinov3_qkvb.eupe_lvd1689m")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--head-convs", type=int, default=2)
    p.add_argument("--fpn-channels", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)

    if args.weights:
        model, _ = load_detector(args.weights, "cpu")
        source = args.weights
    else:
        print("warning: no --weights given; exporting an UNTRAINED model to test the pipeline.")
        model = FireSmokeDetector(backbone_name=args.backbone, pretrained_backbone=False,
                                  image_size=args.imgsz, head="fcos", head_convs=args.head_convs,
                                  fpn_channels=args.fpn_channels)
        source = "untrained"
    model.eval()

    if model.head_type != "fcos":
        raise SystemExit(
            "This checkpoint uses the Faster R-CNN head, which cannot be exported as a batched graph "
            "(measured: batch>1 fails in ONNX Runtime; torch.export fails). Train with --head fcos."
        )

    size = model.config["image_size"]
    out_path = Path(args.output) if args.output else (
        Path(args.weights).with_suffix(".onnx") if args.weights else Path("fire_smoke_fcos.onnx"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"source   : {source}")
    print(f"config   : {model.config['backbone_name']}, {size}px, head fcos x{model.config['head_convs']}, "
          f"pyramid {model.config['fpn_channels']}ch")
    print(f"exporting: {out_path}")

    graph = RawGraph(model).eval()
    torch.onnx.export(
        graph,
        (torch.rand(2, 3, size, size), torch.rand(2, GLOW_FEATURE_DIM)),
        str(out_path),
        dynamo=False,
        opset_version=args.opset,
        input_names=["images", "glow"],
        output_names=OUTPUT_NAMES,
        dynamic_axes={name: {0: "batch"} for name in ["images", "glow", *OUTPUT_NAMES]},
    )
    print(f"exported : {out_path.stat().st_size / 1e6:.1f} MB")

    import onnxruntime as ort

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    failures = []

    # Guarantee there is something to compare: an untrained or confident model
    # can emit nothing above the normal threshold, which would make check 2
    # pass vacuously.
    saved_thresh = model.detector.score_thresh
    model.detector.score_thresh = 0.0

    print("\nverification")
    rng = np.random.default_rng(0)
    for batch in (int(b) for b in args.verify_batches.split(",")):
        images = torch.from_numpy(rng.random((batch, 3, size, size), dtype=np.float32))
        glow = torch.from_numpy(rng.random((batch, GLOW_FEATURE_DIM), dtype=np.float32))

        reference = [t.numpy() for t in graph(images, glow)]
        exported = session.run(None, {"images": images.numpy(), "glow": glow.numpy()})
        tensor_err = max(float(np.abs(a - b).max()) for a, b in zip(reference, exported))

        expected = model(list(images.unbind(0)), glow=glow)[0]
        decoded = model.decode_raw(*exported[:3], image_size=size)
        box_err, score_err, label_ok, count_ok = 0.0, 0.0, True, True
        for e, d in zip(expected, decoded):
            if len(e["boxes"]) != len(d["boxes"]):
                count_ok = False
                continue
            if len(e["boxes"]):
                box_err = max(box_err, float((e["boxes"] - d["boxes"]).abs().max()))
                score_err = max(score_err, float((e["scores"] - d["scores"]).abs().max()))
                label_ok &= bool(torch.equal(e["labels"], d["labels"]))
        n_dets = sum(len(e["boxes"]) for e in expected)

        tensors_pass = tensor_err < 1e-3
        dets_pass = count_ok and label_ok and box_err < 0.5 and score_err < 1e-3
        print(f"  batch {batch}: raw tensors max err {tensor_err:.1e} [{'ok' if tensors_pass else 'FAIL'}]  |  "
              f"{n_dets} detections: box err {box_err:.2e}px, score err {score_err:.1e}, "
              f"labels {'equal' if label_ok else 'DIFFER'}, counts {'equal' if count_ok else 'DIFFER'} "
              f"[{'ok' if dets_pass else 'FAIL'}]")
        if not tensors_pass:
            failures.append(f"batch {batch}: raw tensor mismatch {tensor_err:.2e}")
        if not dets_pass:
            failures.append(f"batch {batch}: decoded detections differ from the PyTorch model")

    model.detector.score_thresh = saved_thresh

    if failures:
        print("\nEXPORT VERIFICATION FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("\nexport verified: batched ONNX reproduces the PyTorch model's detections.")
    print("next: build a TensorRT fp16 engine from this file on the target GPU, e.g.")
    print(f"  trtexec --onnx={out_path.name} --fp16 --minShapes=images:1x3x{size}x{size},glow:1x{GLOW_FEATURE_DIM} "
          f"--optShapes=images:8x3x{size}x{size},glow:8x{GLOW_FEATURE_DIM} "
          f"--maxShapes=images:16x3x{size}x{size},glow:16x{GLOW_FEATURE_DIM} "
          f"--saveEngine={out_path.stem}.engine")


if __name__ == "__main__":
    main()
