import os
import json
import math
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"

# =====================================
# OpenPose JSONフォルダ
# =====================================
json_folder = r"/Users/hanabi/study/ゼミ/自学習/pipe/json.output"

# =====================================
# 保存先
# =====================================
save_path = r"/Users/hanabi/study/ゼミ/自学習/pipe/analyze"

# =====================================
# 動画サイズ
# OpenPose座標を正規化するため
# =====================================
width = 1920
height = 1080

# =====================================
# 距離関数
# =====================================
def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0])**2 +
        (p1[1] - p2[1])**2
    )

# =====================================
# 特徴量保存
# =====================================
features = []

# =====================================
# JSON一覧取得
# =====================================
files = sorted(os.listdir(json_folder))

print("JSON数:", len(files))

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
    # 右手取得
    # =====================================
    hand = data["people"][0]["hand_right_keypoints_2d"]

    # 空データ対策
    if len(hand) == 0:
        continue

    # =====================================
    # 21点へ変換
    # 正規化追加
    # =====================================
    landmarks = []

    for i in range(0, len(hand), 3):

        # 正規化
        x = hand[i] / width
        y = hand[i + 1] / height

        landmarks.append([x, y])

    # =====================================
    # 必要点
    # =====================================
    wrist = landmarks[0]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]

    # =====================================
    # 特徴量
    # =====================================
    d1 = distance(wrist, index_tip)
    d2 = distance(wrist, middle_tip)

    avg_x = sum(p[0] for p in landmarks) / len(landmarks)
    avg_y = sum(p[1] for p in landmarks) / len(landmarks)

    features.append([
        d1,
        d2,
        avg_x,
        avg_y
    ])

# =====================================
# DataFrame
# =====================================
df = pd.DataFrame(
    features,
    columns=[
        "wrist_index",
        "wrist_middle",
        "avg_x",
        "avg_y"
    ]
)

# =====================================
# 標準化
# PCA前に行う
# =====================================
scaler = StandardScaler()

df_scaled = scaler.fit_transform(df)

# =====================================
# PCA
# =====================================
pca = PCA(n_components=2)

result = pca.fit_transform(df_scaled)

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
base_name = os.path.basename(json_folder)

save_dir = os.path.join(
    save_path,
    f"{base_name}_openpose_normalized"
)

os.makedirs(save_dir, exist_ok=True)

# =====================================
# CSV保存
# =====================================
csv_path = os.path.join(
    save_dir,
    f"{base_name}_openpose_normalized.csv"
)

result_df.to_csv(csv_path, index=False)

# =====================================
# 寄与率表示
# =====================================
print("寄与率")
print(pca.explained_variance_ratio_)

print("累積寄与率")
print(sum(pca.explained_variance_ratio_))

# =====================================
# グラフ
# =====================================
plt.figure(figsize=(8,6))

plt.scatter(result[:,0], result[:,1])

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title(f"{base_name} OpenPose PCA")

plt.grid()

# =====================================
# PNG保存
# =====================================
png_path = os.path.join(
    save_dir,
    f"{base_name}_openpose_normalized.png"
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
print("保存完了")
print(csv_path)
print(png_path)