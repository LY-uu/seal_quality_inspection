"""
模型对比评测脚本
=================
在验证集(datasets/visible/val, 100 张,含标注)上对比不同权重与不同标注策略:
  - yolov8_2 : 银边用紧贴框标注训练
  - yolov8_3 : 银边用宽松框标注训练
  - yolov11_1: 银边用紧贴框标注训练(不同模型架构对比)

注意: 当前验证集标注为"紧贴框"口径,因此对"宽松框"训练的模型在 mAP50-95
上会天然吃亏(预测框偏大,与紧贴标注的 IoU 偏低),这是评测口径带来的偏差,
恰好从数值上印证"宽松框会污染回归目标"。
"""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = str(ROOT / "datasets" / "visible" / "data.yaml")
IMGSZ = 640

WEIGHTS = {
    "yolov8_2  (紧贴框)": "models/yolo/weights/yolov8_2.pt",
    "yolov8_3  (宽松框)": "models/yolo/weights/yolov8_3.pt",
    "yolov11_1 (紧贴框)": "models/yolo/weights/yolov11_1.pt",
}


def run_val(name, rel_path):
    p = ROOT / rel_path
    if not p.exists():
        print(f"[跳过] {name} 权重不存在: {rel_path}")
        return None
    model = YOLO(str(p))
    m = model.val(data=DATA, imgsz=IMGSZ, split="val", verbose=False, plots=False)
    box = m.box

    row = {
        "name": name,
        "map50": float(box.map50),
        "map5095": float(box.map),
        "mp": float(box.mp) if hasattr(box, "mp") else None,
        "mr": float(box.mr) if hasattr(box, "mr") else None,
    }

    print(f"\n=== {name} ===")
    print(f"  整体 mAP50={row['map50']:.4f}  mAP50-95={row['map5095']:.4f}"
          f"  P={row['mp']}  R={row['mr']}")

    # 分类别(类别名与 data.yaml 保持一致)
    names = {0: "silver_edge", 1: "foreign_object"}
    idx = list(getattr(box, "ap_class_index", []))
    ap50 = getattr(box, "ap50", None)
    maps = getattr(box, "maps", None)
    for i, ci in enumerate(idx):
        n = names.get(ci, str(ci))
        a50 = ap50[i] if ap50 is not None and len(ap50) > i else float("nan")
        am = maps[i] if maps is not None and len(maps) > i else float("nan")
        print(f"  {n:16s} AP50={a50:.4f}  mAP50-95={am:.4f}")
        row.setdefault("per_class", []).append((n, a50, am))

    return row


def main():
    print(f"数据集: {DATA}")
    print(f"imgsz : {IMGSZ}\n")
    rows = []
    for name, w in WEIGHTS.items():
        r = run_val(name, w)
        if r:
            rows.append(r)

    print("\n\n========== 汇总对比 ==========")
    print(f"{'模型':24s} {'mAP50':>8} {'mAP50-95':>10}")
    for r in rows:
        print(f"{r['name']:24s} {r['map50']:8.4f} {r['map5095']:10.4f}")


if __name__ == "__main__":
    main()
