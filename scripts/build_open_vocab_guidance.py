import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image


def _load_records(path: str) -> list[dict[str, Any]]:
    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    with open(path, "r", encoding="utf-8") as fin:
        payload = json.load(fin)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "records" in payload and isinstance(payload["records"], list):
        return payload["records"]
    raise ValueError("Input manifest must be json/jsonl list, or {'records': [...]} format.")


def _normalize_keywords(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [x.strip() for x in value.split(",")]
        return [x for x in parts if x]
    if isinstance(value, list):
        out = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    return []


def _resolve_image_path(item: dict[str, Any], image_root: str | None) -> str | None:
    for key in ("image", "image_path", "img_path", "path"):
        if key in item:
            p = str(item[key])
            if image_root and not os.path.isabs(p):
                p = os.path.join(image_root, p)
            if os.path.exists(p):
                return p
    return None


@dataclass
class DetectResult:
    boxes: list[list[int]]
    scores: list[float]


class OWLv2Detector:
    def __init__(self, model_name: str, device: str, threshold: float):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.threshold = threshold

    @torch.no_grad()
    def predict(self, image: Image.Image, query: str) -> DetectResult:
        inputs = self.processor(text=[query], images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        post = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=float(self.threshold),
            target_sizes=target_sizes,
        )[0]
        boxes = post["boxes"].detach().cpu().numpy().tolist()
        scores = post["scores"].detach().cpu().numpy().tolist()
        boxes = [[int(round(v)) for v in box] for box in boxes]
        scores = [float(s) for s in scores]
        return DetectResult(boxes=boxes, scores=scores)


class SAMSegmenter:
    def __init__(self, model_name: str, device: str):
        from transformers import SamModel, SamProcessor

        self.processor = SamProcessor.from_pretrained(model_name)
        self.model = SamModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def predict(self, image: Image.Image, boxes_xyxy: list[list[int]]) -> tuple[np.ndarray | None, float | None]:
        if not boxes_xyxy:
            return None, None
        inputs = self.processor(
            images=image,
            input_boxes=[boxes_xyxy],
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
        iou_scores = outputs.iou_scores.detach().cpu().numpy()[0]
        # Flatten candidate masks and select the best iou one.
        cand_masks = []
        cand_scores = []
        for i in range(masks.shape[0]):
            for j in range(masks.shape[1]):
                cand_masks.append(masks[i, j].numpy())
                cand_scores.append(float(iou_scores[i, j]))
        if not cand_masks:
            return None, None
        idx = int(np.argmax(np.asarray(cand_scores)))
        best = (cand_masks[idx] > 0).astype(np.uint8)
        return best, float(cand_scores[idx])


def _resize_mask(mask: np.ndarray, max_side: int) -> np.ndarray:
    if max(mask.shape[0], mask.shape[1]) <= max_side:
        return mask
    h, w = mask.shape
    scale = float(max_side) / float(max(h, w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    pil = Image.fromarray((mask * 255).astype(np.uint8))
    resized = pil.resize((nw, nh), Image.NEAREST)
    return (np.array(resized) > 127).astype(np.uint8)


def build_guidance(
    records: list[dict[str, Any]],
    image_root: str | None,
    detector: OWLv2Detector | None,
    segmenter: SAMSegmenter | None,
    max_mask_side: int,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(records):
        image_path = _resolve_image_path(item, image_root=image_root)
        if image_path is None:
            continue
        keywords = _normalize_keywords(item.get("keywords"))
        if not keywords:
            continue
        image = Image.open(image_path).convert("RGB")
        image_key = os.path.basename(image_path)
        output.setdefault(image_key, {})
        for keyword in keywords:
            if detector is None:
                det = DetectResult(boxes=[], scores=[])
            else:
                det = detector.predict(image=image, query=keyword)

            seg_mask = None
            seg_score = None
            if segmenter is not None and det.boxes:
                seg_mask, seg_score = segmenter.predict(image=image, boxes_xyxy=det.boxes[:1])
                if seg_mask is not None:
                    seg_mask = _resize_mask(seg_mask, max_side=max_mask_side)

            output[image_key][keyword.lower()] = {
                "boxes": det.boxes,
                "scores": det.scores,
                "mask": seg_mask.tolist() if seg_mask is not None else None,
                "seg_score": float(seg_score) if seg_score is not None else None,
                "meta": {
                    "record_index": idx,
                    "image_path": image_path,
                },
            }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build open-vocab guidance json for qwen2_vl auto head mining.")
    parser.add_argument("--input", required=True, help="Input manifest path (.json/.jsonl).")
    parser.add_argument("--output", required=True, help="Output guidance json path.")
    parser.add_argument("--image-root", default=None, help="Optional root path for relative image paths.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--detector", default="owlv2", choices=["owlv2", "none"])
    parser.add_argument("--det-model", default="google/owlv2-base-patch16-ensemble")
    parser.add_argument("--det-threshold", type=float, default=0.15)

    parser.add_argument("--segmenter", default="sam", choices=["sam", "none"])
    parser.add_argument("--seg-model", default="facebook/sam-vit-base")
    parser.add_argument("--mask-max-side", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    records = _load_records(args.input)

    detector = None
    if args.detector == "owlv2":
        detector = OWLv2Detector(
            model_name=args.det_model,
            device=args.device,
            threshold=args.det_threshold,
        )

    segmenter = None
    if args.segmenter == "sam":
        segmenter = SAMSegmenter(
            model_name=args.seg_model,
            device=args.device,
        )

    guidance = build_guidance(
        records=records,
        image_root=args.image_root,
        detector=detector,
        segmenter=segmenter,
        max_mask_side=args.mask_max_side,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        json.dump(guidance, fout, ensure_ascii=False, indent=2)
    print(f"Saved guidance to {args.output}, image_count={len(guidance)}")


if __name__ == "__main__":
    main()
