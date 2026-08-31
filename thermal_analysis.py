"""
红外封口区域温度分析(可行性验证)
=====================================
思路:
1. 伪彩红外图(蓝冷红热),热信号取 heat = R - B;
2. 行均值投影 -> 定位封口条带的垂直位置与厚度;
3. 在封口行内做列均值投影 -> 定位水平范围(长度);
4. 对定位出的封口区域计算温度统计量(均值/方差/最值/温差);
5. 以前 REF_COUNT 张"标准图"建立基准,用 z-score 衡量每张图的偏离程度。

无标签,故不输出硬性 NG/OK,只输出偏离分数与异常提示,用于验证算法可行性。
"""

import os
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent
IR_DIR = ROOT / "datasets" / "infrared"
OUT_DIR = ROOT / "results" / "infrared_analysis"

REF_COUNT = 6          # 前 N 张作为"标准/参考"
WARM_THRESH = 0.0      # 暖区阈值:行均值(R-B) >= 0 的行算封口条带(上缓坡需放宽)
HALF_H_MAX = 30        # 封口条带垂直半厚度上限(像素,绝对保险)
WIDTH_RATIO = 0.85     # 水平先验宽度占比(封口约占图像宽度 0.8~0.9)
Z_THRESH = 3.0         # z-score 超过该值提示异常
Y_TOL = 5.0            # 封口垂直位置绝对容差(行),位置微偏不判异常


def load_heat(path):
    """读取红外图,返回热信号 heat = R - B(float32)。"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取 {path}")
    img = img.astype(np.float32)
    if img.ndim == 3:
        return img[:, :, 2] - img[:, :, 0]
    return img


def _expand(mask, center):
    """从 center 向两侧扩展,返回包含 center 的最大连续 True 区间 [l, r)。"""
    n = len(mask)
    l = center
    while l - 1 >= 0 and mask[l - 1]:
        l -= 1
    r = center
    while r < n and mask[r]:
        r += 1
    return l, r


def locate_seal(heat):
    """
    定位封口区域。
    返回 (bbox, region, row_mean, col_mean, peak_val, y_c)
    bbox = (x1, y1, x2, y2)
    策略:
      - 垂直方向:行均值投影定位中心 + 阈值扩展带宽(限幅);
      - 水平方向:封口长度接近图像宽度,用先验宽度,仅用列投影定中心(热区质心)。
    """
    H, W = heat.shape

    # 1) 行投影:定位垂直位置与厚度
    row_mean = heat.mean(axis=1)
    y_c = int(np.argmax(row_mean))
    peak_val = float(row_mean[y_c])
    if peak_val <= 0:
        raise ValueError("未找到正热信号,图像可能异常")

    row_mask = row_mean >= WARM_THRESH
    y1, y2 = _expand(row_mask, y_c)
    # 限幅,避免把背景带进来(条带为上方缓坡、下方陡崖)
    y1 = max(y1, y_c - HALF_H_MAX)
    y2 = min(y2, y_c + HALF_H_MAX)

    # 2) 水平:列投影定中心(热区质心),宽度用先验
    col_mean = heat[y1:y2, :].mean(axis=0)
    pos = np.where(col_mean > 0)[0]
    if len(pos) > 0:
        x_c = int(round(np.average(pos, weights=col_mean[pos])))
    else:
        x_c = W // 2
    half_w = int(round(W * WIDTH_RATIO / 2))
    x1 = min(max(x_c - half_w, 0), W - 2 * half_w)
    x2 = x1 + 2 * half_w

    bbox = (x1, y1, x2, y2)
    region = heat[y1:y2, x1:x2]
    return bbox, region, row_mean, col_mean, peak_val, y_c


def region_stats(region):
    """封口区域温度统计量。"""
    return {
        "mean": float(region.mean()),   # 平均热强度
        "std": float(region.std()),     # 均匀性
        "max": float(region.max()),
        "min": float(region.min()),
        "span": float(region.max() - region.min()),  # 温差
        "area": int(region.size),
    }


def extract_features(path):
    """对单张图提取用于判定的特征。"""
    heat = load_heat(path)
    bbox, region, row_mean, col_mean, peak_val, y_c = locate_seal(heat)
    s = region_stats(region)
    x1, y1, x2, y2 = bbox
    feat = {
        "y_c": float(y_c),          # 封口垂直位置
        "peak_val": peak_val,       # 行投影峰值(封口条带平均热强度)
        "mean": s["mean"],          # 封口区域平均热强度
        "std": s["std"],            # 封口区域温度均匀性
        "height": float(y2 - y1),   # 封口条带厚度
    }
    return heat, bbox, region, s, feat


def zscore_features(features):
    """以参考图(前 REF_COUNT 张)为基准,计算每张图各特征的 z-score 与综合偏离分数。"""
    ref_idx = list(range(REF_COUNT))
    ref = {k: np.array([features[i][k] for i in ref_idx], dtype=np.float64)
           for k in features[0]}
    mu = {k: v.mean() for k, v in ref.items()}
    sd = {k: v.std() for k, v in ref.items()}

    rows = []
    for i, f in enumerate(features):
        zs = {}
        for k in f:
            denom = sd[k] if sd[k] > 1e-6 else 1.0
            zs[k] = abs(f[k] - mu[k]) / denom
        # 温度相关特征综合取最大 z(位置 y_c 改用绝对容差,不参与 score)
        score = max(zs["peak_val"], zs["mean"])
        rows.append({"idx": i, "z": zs, "score": score})
    return mu, sd, rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(IR_DIR.glob("*.tif"))
    files = [f for f in files if "ACamera" in f.name]   # 暂不考虑 DCamera(另一路相机/工位)
    print(f"红外图总数: {len(files)}")
    print(f"参考基准: 前 {REF_COUNT} 张\n")

    features, metas = [], []
    for i, f in enumerate(files):
        heat, bbox, region, s, feat = extract_features(f)
        features.append(feat)
        metas.append((f.name, heat, bbox, s))

    mu, sd, rows = zscore_features(features)

    # 打印特征表与判定
    header = (f"{'idx':>3} {'file':28s} {'y_c':>6} {'peak':>7} {'mean':>7} "
              f"{'std':>6} {'h':>4} {'score':>6}  flag")
    print(header)
    flags = []
    for i, (name, heat, bbox, s) in enumerate(metas):
        zs = rows[i]["z"]
        score = rows[i]["score"]
        f = features[i]
        # 位置 y_c 用绝对容差,温度特征用 z-score
        why = []
        if abs(f["y_c"] - mu["y_c"]) > Y_TOL:
            why.append("y_c")
        for k in ("peak_val", "mean", "std"):
            if zs[k] > Z_THRESH:
                why.append(k)
        flag = ("异常:" + "+".join(why)) if why else ""
        if why:
            flags.append(i)
        print(f"{i:3d} {name:28s} {f['y_c']:6.1f} {f['peak_val']:7.1f} {f['mean']:7.1f} "
              f"{f['std']:6.1f} {f['height']:4.0f} {score:6.2f}  {flag}")

    print(f"\n参考基准(前{REF_COUNT}张) 均值±标准差:")
    for k in mu:
        print(f"  {k:10s}: {mu[k]:8.1f} ± {sd[k]:6.1f}")

    print(f"\n被标记异常: {len(flags)} 张 -> {[metas[i][0] for i in flags]}")

    # 保存可视化与结果
    for i, (name, heat, bbox, s) in enumerate(metas):
        # 用原始 RGB 图叠加(读回原图)
        rgb = cv2.imread(str(IR_DIR / name), cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"mean={s['mean']:.0f} std={s['std']:.0f}"
        cv2.putText(rgb, label, (x1, max(10, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        Image.fromarray(rgb).save(OUT_DIR / f"{i:02d}_{Path(name).stem}.png")

    result = {
        "reference_indices": list(range(REF_COUNT)),
        "reference": {k: {"mean": mu[k], "std": sd[k]} for k in mu},
        "files": [{"idx": i, "name": m[0], "features": features[i],
                   "z": rows[i]["z"], "score": rows[i]["score"]}
                  for i, m in enumerate(metas)],
    }
    with open(OUT_DIR / "report.json", "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)

    print(f"\n可视化结果已保存到: {OUT_DIR}")
    print(f"报告已保存到: {OUT_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
