import os, glob, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- ここだけ変えればOK ---
FOLDER = "demo2.json"   # OpenPoseの出力フォルダ名
OUT_CSV = "demo1_handR_wrist_finger.csv"
OUT_PNG = "demo1_finger_wrist_ratio.png"

# hand_right_keypoints_2d の点番号
WRIST_IDX = 0   # hand_right_0（手首）
FINGER_IDX = 8  # hand_right_8（人差し指先端）

def read_hand_point(handR, idx):
    """idx番の(x,y,confidence)を返す"""
    x = handR[idx * 3 + 0]
    y = handR[idx * 3 + 1]
    c = handR[idx * 3 + 2]
    return x, y, c

def main():
    folder_path = os.path.join(".", FOLDER)
    json_files = sorted(glob.glob(os.path.join(folder_path, "*keypoints.json")))

    if not json_files:
        raise FileNotFoundError(f"JSONが見つかりません: {folder_path}")

    rows = []
    prev_wrist = None
    prev_finger = None
    total_wrist = 0.0
    total_finger = 0.0

    for i, jf in enumerate(json_files):
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        people = data.get("people", [])
        if not people:
            continue  # 人が検出されないフレームはスキップ

        handR = people[0].get("hand_right_keypoints_2d", [])
        if len(handR) < (max(WRIST_IDX, FINGER_IDX) + 1) * 3:
            continue  # 右手情報が不足

        wx, wy, wc = read_hand_point(handR, WRIST_IDX)
        fx, fy, fc = read_hand_point(handR, FINGER_IDX)

        # confidenceが低いフレームは除外（閾値は必要なら調整）
        if wc < 0.2 or fc < 0.2:
            continue

        wrist = (wx, wy)
        finger = (fx, fy)

        dw = df = np.nan
        if prev_wrist is not None and prev_finger is not None:
            dw = math.dist(wrist, prev_wrist)
            df = math.dist(finger, prev_finger)
            total_wrist += dw
            total_finger += df

        rows.append({
            "frame_index": i,
            "wrist_x": wx, "wrist_y": wy, "wrist_conf": wc,
            "finger_x": fx, "finger_y": fy, "finger_conf": fc,
            "wrist_step_dist(px)": dw,
            "finger_step_dist(px)": df,
        })

        prev_wrist = wrist
        prev_finger = finger

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV出力: {OUT_CSV}  rows={len(df)}")

    total = total_wrist + total_finger
    wrist_ratio = (total_wrist / total * 100) if total > 0 else np.nan
    finger_ratio = (total_finger / total * 100) if total > 0 else np.nan

    print(f"finger_total(px)={total_finger:.2f}, wrist_total(px)={total_wrist:.2f}")
    print(f"finger_ratio(%)={finger_ratio:.2f}, wrist_ratio(%)={wrist_ratio:.2f}")

    # --- グラフ ---
    plt.rcParams["font.family"] = "MS Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    labels = ["指先", "手首"]
    vals = [finger_ratio, wrist_ratio]

    plt.figure(figsize=(7, 4))
    y = np.arange(len(labels))
    plt.barh(y, vals)
    plt.xlim(0, 100)
    plt.xlabel("寄与率（%）")
    plt.yticks(y, labels)
    for i, v in enumerate(vals):
        plt.text(v + 1, i, f"{v:.1f}%", va="center")
    plt.title("demo1：指先・手首の寄与比率（移動距離ベース）")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.show()
    print(f"[OK] グラフ出力: {OUT_PNG}")

if __name__ == "__main__":
    main()
