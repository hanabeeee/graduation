import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ===============================
# 設定
# ===============================
MP_JSON = r"C:\Users\student\Desktop\2026\pipe\json.output\1_1.json"

OUT_CSV = r"C:\Users\student\Desktop\2026\pipe\pca"
OUT_PNG = r"C:\Users\student\Desktop\2026\pipe\pca"

TOPK = 20

# ===============================
# JSON読み込み
# ===============================
with open(MP_JSON, "r") as f:
    data = json.load(f)

# ===============================
# 特徴量抽出（dy）
# ===============================
rows = []
prev_y = None

for frame in data:

    landmarks = frame.get("landmarks", [])

    if len(landmarks) == 0:
        continue

    pts = np.array(landmarks)

    # y座標だけ取り出す
    y_vals = pts[:, 1]

    # 最初のフレームはスキップ
    if prev_y is None:
        prev_y = y_vals
        continue

    # dy計算
    dy = y_vals - prev_y

    rows.append(dy)

    prev_y = y_vals

# DataFrame化
df_feat = pd.DataFrame(
    rows,
    columns=[f"landmark_{i}" for i in range(21)]
)

#jsonファイル名からcsvファイル名を作成
base_name =os.path.splitext(os.path.basename(MP_JSON))[0]
SAVE = os.path.join(OUT_CSV, f"{base_name}")
os.makedirs(SAVE, exist_ok=True)

OUT_CSV = os.path.join(SAVE, f"{base_name}.csv")


# CSV保存
df_feat.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print(f"[OK] CSV保存: {OUT_CSV}")

# ===============================
# PCA
# ===============================
X = df_feat.values

# 標準化
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# PCA
pca = PCA()
pca.fit(Xs)

# ===============================
# PC1寄与率
# ===============================
loadings = pca.components_[0]

contrib = loadings ** 2

contrib_ratio = contrib / contrib.sum()

result = pd.Series(
    contrib_ratio,
    index=df_feat.columns
).sort_values(ascending=False)

top = result.head(TOPK) * 100

# ===============================
# グラフ
# ===============================
plt.rcParams["font.family"] = "Meiryo"

plt.figure(figsize=(9,6))

plt.barh(top.index[::-1], top.values[::-1])

plt.xlabel("寄与率 (%)")
plt.ylabel("ランドマーク")
plt.title("MediaPipe PCA PC1寄与率")

for i, v in enumerate(top.values[::-1]):
    plt.text(v + 0.3, i, f"{v:.1f}%", va="center")

plt.tight_layout()

##ファイル名の作成
OUT_PNG = os.path.join(SAVE, f"{base_name}.png")

plt.savefig(OUT_PNG, dpi=300)

plt.show()

print(f"[OK] PNG保存: {OUT_PNG}")

# ===============================
# 分散説明率
# ===============================
print("\n=== 分散説明率 ===")

for i, r in enumerate(pca.explained_variance_ratio_[:5], start=1):
    print(f"PC{i}: {r*100:.2f}%")
