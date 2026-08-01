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

json_path = "/Users/hanabi/study/ゼミ/自学習/pipe/json.output/1_1.json"
save_path = "/Users/hanabi/study/ゼミ/自学習/PCAhikaku"

base_name = os.path.splitext(os.path.basename(json_path))[0]

save_dir = os.path.join(
    save_path,
    f"{base_name}"
)

os.makedirs(save_dir, exist_ok=True)

with open(json_path, "r") as f:
    data = json.load(f)

# =========================
# JSON → NumPy配列変換
# =========================

all_frames = []

for frame in data:

    frame_vector = []

    # 手首を基準に相対座標化
    wrist_x = frame["landmarks"][0][0]
    wrist_y = frame["landmarks"][0][1]

    for landmark in frame["landmarks"]:

        x = landmark[0]
        y = landmark[1]

        # 相対座標化
        x = x - wrist_x
        y = y - wrist_y

        # x,yを1列に並べる
        frame_vector.extend([x, y])

    all_frames.append(frame_vector)

# NumPy配列化
X = np.array(all_frames)

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