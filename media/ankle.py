import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === 入力 ===
csv_path = "right_ankle_y.csv"   # さっき作ったCSVのパス
CM_PER_PX = 100.0 / 294.027      # 白線1m(100cm) / 294.027px

# === 読み込み ===
df = pd.read_csv(csv_path)

# フレーム列を推定 or 生成
frame_col = next((c for c in df.columns if 'frame' in c.lower()), None)
if frame_col is None:
    frame_col = 'frame'
    df[frame_col] = np.arange(len(df))

# y[cm] or y[px] を探す
y_cm_col = next((c for c in df.columns if ('cm' in c.lower() and ('ankle' in c.lower() or 'y' in c.lower()))), None)
y_px_col = next((c for c in df.columns if ('px' in c.lower() and ('ankle' in c.lower() or 'y' in c.lower()))), None)

# cm列が無ければpx→cm変換
if y_cm_col is None:
    if y_px_col is None:
        raise ValueError("足首Yの列が見つかりません。CSVの列名を教えてください。")
    df['right_ankle_y(cm)'] = df[y_px_col] * CM_PER_PX
    y_cm_col = 'right_ankle_y(cm)'

# 欠損値を少しだけ平滑化（オプション）
y_cm = pd.Series(df[y_cm_col]).astype(float)
y_cm_i = y_cm.interpolate(limit_direction='both')
df[y_cm_col] = y_cm_i

# サマリ
summary = {
    'frames': len(df),
    'min_y_cm': float(np.nanmin(df[y_cm_col])),
    'max_y_cm': float(np.nanmax(df[y_cm_col])),
    'range_cm': float(np.nanmax(df[y_cm_col]) - np.nanmin(df[y_cm_col])),
    'mean_y_cm': float(np.nanmean(df[y_cm_col]))
}
print("Summary:", summary)

# 図を保存
plt.figure()
plt.plot(df[frame_col], df[y_cm_col])
plt.xlabel("Frame")
plt.ylabel("Right ankle Y (cm)")
plt.title("Right ankle height over frames")
plt.tight_layout()
out_plot = Path("right_ankle_y_plot.png")
plt.savefig(out_plot)
print(f"Saved plot to {out_plot.resolve()}")

# クリーンCSV保存
out_csv = Path("right_ankle_y_clean.csv")
df[[frame_col, y_cm_col]].to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"Saved cleaned CSV to {out_csv.resolve()}")
