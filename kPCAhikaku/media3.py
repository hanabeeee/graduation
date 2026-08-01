import os
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# =========================
# JSONファイル読み込み
# =========================

json_path = "/Users/hanabi/study/ゼミ/自学習/pipe/json.output/2026_2.json"
save_path = "/Users/hanabi/study/ゼミ/自学習/kPCAhikaku"

base_name = os.path.splitext(os.path.basename(json_path))[0]

save_dir = os.path.join(
    save_path,
    f"{base_name}"
)

os.makedirs(save_dir, exist_ok=True)

with open(json_path, "r") as f:
    data = json.load(f)

# =========================
# 欠損補完関数
# =========================

def is_missing(landmark):

    # x,y,z全部0なら欠損扱い
    return (
        landmark[0] == 0 and
        landmark[1] == 0 and
        landmark[2] == 0
    )

def interpolate_landmark(data, frame_idx, landmark_idx):

    prev_landmark = None
    next_landmark = None

    # ===== 前フレーム探索 =====
    for i in range(frame_idx - 1, -1, -1):

        lm = data[i]["landmarks"][landmark_idx]

        if not is_missing(lm):
            prev_landmark = lm
            break

    # ===== 後フレーム探索 =====
    for i in range(frame_idx + 1, len(data)):

        lm = data[i]["landmarks"][landmark_idx]

        if not is_missing(lm):
            next_landmark = lm
            break

    # ===== 補完 =====

    # 前後両方ある場合
    if prev_landmark is not None and next_landmark is not None:

        return [
            (prev_landmark[0] + next_landmark[0]) / 2,
            (prev_landmark[1] + next_landmark[1]) / 2,
            (prev_landmark[2] + next_landmark[2]) / 2
        ]

    # 前だけある
    elif prev_landmark is not None:

        return prev_landmark

    # 後だけある
    elif next_landmark is not None:

        return next_landmark

    # 全部欠損
    else:

        return [0, 0, 0]

# =========================
# OpenPoseと共通の関節
# =========================
# Nose, LShoulder, RShoulder,
# LElbow, RElbow, LWrist, RWrist
common_indices = [0, 11, 12, 13, 14, 15, 16]

all_frames = []

for frame_idx, frame in enumerate(data):

    frame_vector = []

    # =========================
    # 肩中心を基準に相対座標化
    # =========================

    left_shoulder = frame["landmarks"][11]
    right_shoulder = frame["landmarks"][12]

    # 欠損補完
    if is_missing(left_shoulder):

        left_shoulder = interpolate_landmark(
            data,
            frame_idx,
            11
        )

    if is_missing(right_shoulder):

        right_shoulder = interpolate_landmark(
            data,
            frame_idx,
            12
        )

    center_x = (
        left_shoulder[0]
        + right_shoulder[0]
    ) / 2

    center_y = (
        left_shoulder[1]
        + right_shoulder[1]
    ) / 2

    for idx in common_indices:

        landmark = frame["landmarks"][idx]

        # 欠損なら補完
        if is_missing(landmark):

            landmark = interpolate_landmark(
                data,
                frame_idx,
                idx
            )

        x = landmark[0]
        y = landmark[1]

        # 相対座標化
        x = x - center_x
        y = y - center_y

        # x,yを1列に並べる
        frame_vector.extend([x, y])

    all_frames.append(frame_vector)

# =========================
# NumPy配列化
# =========================

X = np.array(all_frames)

print()
print("===== 共通関節 =====")
print("使用関節:")
print(common_indices)

# =========================
# CSV保存
# =========================

df = pd.DataFrame(X)

csv_path = os.path.join(
    save_dir,
    f"{base_name}_media.csv"
)

df.to_csv(csv_path, index=False)

# =========================
# 配列情報表示
# =========================

print("===== データ情報 =====")
print("配列サイズ:", X.shape)
print()

print("先頭5フレーム:")
print(X[:5])

# =========================
# 標準化
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================
# PCA実行
# =========================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# =========================
# 寄与率表示
# =========================

print()
print("===== PCA結果 =====")
print("主成分後サイズ:", X_pca.shape)
print()

print("寄与率:")
print(pca.explained_variance_ratio_)

print()
print("累積寄与率:")
print(np.sum(pca.explained_variance_ratio_))

# =========================
# PCA散布図
# =========================

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=range(len(X_pca)),
    cmap="viridis",
    s=15
)

# 時系列を線で接続
plt.plot(
    X_pca[:, 0],
    X_pca[:, 1],
    alpha=0.3
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.axis("equal")

plt.title(
    f"MediaPipe PCA\n"
    f"PC1={pca.explained_variance_ratio_[0]:.2f}, "
    f"PC2={pca.explained_variance_ratio_[1]:.2f}"
)

plt.colorbar(scatter, label="Frame")

plt.grid()

png_path = os.path.join(
    save_dir,
    f"{base_name}_media.png"
)

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()