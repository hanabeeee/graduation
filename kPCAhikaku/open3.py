import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# =====================================
# JSONフォルダ
# =====================================
json_folder = r"/Users/hanabi/study/ゼミ/自学習/openpose/2026_2.json"

# =====================================
# 保存先
# =====================================
save_path = r"/Users/hanabi/study/ゼミ/自学習/kPCAhikaku"

# =====================================
# 特徴量保存
# =====================================
all_frames = []

# =====================================
# JSON一覧取得
# =====================================
files = sorted(os.listdir(json_folder))

print("JSON数:", len(files))

# =====================================
# MediaPipeと共通の関節
# =====================================
# Nose, RShoulder, RElbow, RWrist,
# LShoulder, LElbow, LWrist
common_indices = [0, 2, 3, 4, 5, 6, 7]

# =====================================
# 全JSON解析
# =====================================
for file in files:

    if not file.endswith(".json"):
        continue

    path = os.path.join(json_folder, file)

    with open(path, "r") as f:
        data = json.load(f)

    # 人物なし
    if len(data["people"]) == 0:
        continue

    # =====================================
    # OpenPose座標取得
    # =====================================
    # pose_keypoints_2d
    # [x1,y1,c1,x2,y2,c2,...]
    keypoints = data["people"][0]["pose_keypoints_2d"]

    frame_vector = []

    # 首(1)を基準に相対座標化
    neck_x = keypoints[3]
    neck_y = keypoints[4]

    # 共通関節のみ使用
    for idx in common_indices:

        base = idx * 3

        x = keypoints[base]
        y = keypoints[base + 1]
        c = keypoints[base + 2]

        # 信頼度が低い点を除外
        if c < 0.1:
            x = 0
            y = 0
        else:
            # 相対座標化
            x = x - neck_x
            y = y - neck_y

        # confidenceは使わない
        frame_vector.extend([x, y])

    all_frames.append(frame_vector)

# =====================================
# NumPy配列化
# =====================================
X = np.array(all_frames)

print()
print("===== 共通関節 =====")
print(common_indices)

# =====================================
# DataFrame
# =====================================
df = pd.DataFrame(X)

# =====================================
# 標準化
# =====================================
scaler = StandardScaler()

X_scaled = scaler.fit_transform(df)

# =====================================
# PCA
# =====================================
pca = PCA(n_components=2)

result = pca.fit_transform(X_scaled)

# =====================================
# PCA結果DataFrame
# =====================================
result_df = pd.DataFrame(
    result,
    columns=["PC1", "PC2"]
)

# =====================================
# フォルダ作成
# =====================================
base_name = os.path.splitext(os.path.basename(json_folder))[0]

save_dir = os.path.join(
    save_path,
    f"{base_name}"
)

os.makedirs(save_dir, exist_ok=True)

# =====================================
# CSV保存
# =====================================
csv_path = os.path.join(
    save_dir,
    f"{base_name}_openpose.csv"
)

result_df.to_csv(csv_path, index=False)

# =====================================
# PCA結果表示
# =====================================
print()
print("===== PCA結果 =====")

print("主成分後サイズ:")
print(result.shape)

print()

print("寄与率:")
print(pca.explained_variance_ratio_)

print()

print("累積寄与率:")
print(np.sum(pca.explained_variance_ratio_))

# =====================================
# グラフ
# =====================================
plt.figure(figsize=(8,6))

scatter = plt.scatter(
    result[:,0],
    result[:,1],
    c=range(len(result)),
    cmap="viridis",
    s=15
)

# 時系列を線で接続
plt.plot(
    result[:,0],
    result[:,1],
    alpha=0.3
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.axis("equal")

plt.title(
    f"{base_name} OpenPose PCA\n"
    f"PC1={pca.explained_variance_ratio_[0]:.2f}, "
    f"PC2={pca.explained_variance_ratio_[1]:.2f}"
)

plt.colorbar(scatter, label="Frame")

plt.grid()

# =====================================
# PNG保存
# =====================================
png_path = os.path.join(
    save_dir,
    f"{base_name}_openpose.png"
)

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================
# 保存確認
# =====================================
print()
print("保存完了")
print(csv_path)
print(png_path)