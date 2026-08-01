import json
import glob
import pandas as pd

# OpenPoseの出力jsonが並んでいるフォルダを指定
files = sorted(glob.glob("hokou1.json/*.json"))

RIGHT_ANKLE = 11  # BODY_25の場合
y_list = []

for i, f in enumerate(files):
    with open(f, "r") as js:
        data = json.load(js)
    if not data["people"]:
        y_list.append(None)
        continue

    kp = data["people"][0]["pose_keypoints_2d"]
    y = kp[RIGHT_ANKLE * 3 + 1]
    conf = kp[RIGHT_ANKLE * 3 + 2]

    # 信頼度が低いフレームは無視
    if conf < 0.2:
        y = None
    y_list.append(y)

# 結果をDataFrameで整理
df = pd.DataFrame({
    "frame": list(range(len(y_list))),
    "right_ankle_y(px)": y_list
})

# ピクセルをcmに変換（白線=294.027px = 100cm）
cm_per_px = 100 / 294.027
df["right_ankle_y(cm)"] = df["right_ankle_y(px)"] * cm_per_px

# 結果をCSVで保存
df.to_csv("right_ankle_y.csv", index=False, encoding="utf-8-sig")
print("✅ right_ankle_y.csv に保存しました。")
