import cv2
import numpy as np
import os

# ======================================
# 設定
# ======================================
IMAGE_PATH = "finnger/media/hand3.png"  # ※撮影した画像のパスに変更してください
RESULT_DIR = "result"
os.makedirs(RESULT_DIR, exist_ok=True)

# 印刷したArUcoマーカーの1辺の実際の長さ(cm)を設定
MARKER_LENGTH_CM = 5.0  

# ======================================
# 1. 画像読込と準備
# ======================================
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"エラー: 画像 '{IMAGE_PATH}' が見つかりません。パスを確認してください。")
    exit()

image_draw = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ======================================
# 2. ArUcoマーカー検出とスケール計算 (1ピクセル = 何cmか)
# ======================================
# ※一般的なDICT_4X4_50を使用（使用した辞書に合わせて変更してください）
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# OpenCVのバージョンによる互換性対応
try:
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, rejected = detector.detectMarkers(image)
except AttributeError:
    # OpenCV 4.6以前の古いバージョンの場合
    corners, ids, rejected = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)

cm_per_pixel = 1.0  # デフォルト値
marker_center = None

if ids is not None and len(ids) > 0:
    # マーカーの4つの角の座標を取得
    marker_corners = corners[0][0]
    
    # 4辺のピクセル長さを計算して平均をとる
    w1 = np.linalg.norm(marker_corners[0] - marker_corners[1])
    w2 = np.linalg.norm(marker_corners[1] - marker_corners[2])
    w3 = np.linalg.norm(marker_corners[2] - marker_corners[3])
    w4 = np.linalg.norm(marker_corners[3] - marker_corners[0])
    marker_width_px = (w1 + w2 + w3 + w4) / 4.0
    
    # 1ピクセルあたりの長さ(cm)を算出
    cm_per_pixel = MARKER_LENGTH_CM / marker_width_px
    
    # マーカーの中心座標を算出（誤検出防止の除外処理に使用）
    cx = int(np.mean(marker_corners[:, 0]))
    cy = int(np.mean(marker_corners[:, 1]))
    marker_center = (cx, cy)
    
    # 画像にマーカー枠を描画
    cv2.aruco.drawDetectedMarkers(image_draw, corners, ids)
    print(f"[SUCCESS] マーカー検出成功: 1px = {cm_per_pixel:.4f} cm")
else:
    print("[WARNING] マーカーが検出されませんでした。計測値はピクセル単位の比率になります。")

# ======================================
# 3. 前処理 (ノイズ除去・二値化・穴埋め)
# ======================================
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((5, 5), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

# ======================================
# 4. 手の最大輪郭を取得 (マーカーは除外)
# ======================================
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(contours) == 0:
    print("[ERROR] 輪郭が見つかりません")
    exit()

largest_contour = None
max_area = 0

for c in contours:
    # 輪郭の中にマーカーの中心が含まれている場合は「マーカーの輪郭」なので除外
    if marker_center is not None:
        if cv2.pointPolygonTest(c, marker_center, False) >= 0:
            continue
            
    area = cv2.contourArea(c)
    if area > max_area:
        max_area = area
        largest_contour = c

if largest_contour is None:
    print("[ERROR] 手の輪郭が見つかりません")
    exit()

# 手の輪郭を緑色で描画
cv2.drawContours(image_draw, [largest_contour], -1, (0, 255, 0), 2)

# ======================================
# 5. 【指標A・B・C】 の計測と描画
# ======================================

# --- 【指標 A】 手の投影面積（接触面積の代替） ---
area_px = cv2.contourArea(largest_contour)
A_area_cm2 = area_px * (cm_per_pixel ** 2)

# --- 【指標 B】 手幅（最小外接矩形） ---
rect = cv2.minAreaRect(largest_contour)
box = cv2.boxPoints(rect)
box = np.int32(box)

(rect_cx, rect_cy), (rect_w, rect_h), angle = rect
# 矩形の縦・横のうち、短い方を「手幅(B)」とする
B_width_px = min(rect_w, rect_h)
B_width_cm = B_width_px * cm_per_pixel

# 手幅の矩形を青色で描画
cv2.drawContours(image_draw, [box], 0, (255, 0, 0), 2)

# --- 【指標 C】 「コ」の字の大きさ（親指〜人差し指の間隔） ---
hull = cv2.convexHull(largest_contour)

fingertips = []
for pt in hull:
    pt = pt[0]
    # 近すぎる点を間引く（30ピクセル以上離れているものだけ指先候補とする）
    if all(np.linalg.norm(pt - np.array(f)) > 30 for f in fingertips):
        fingertips.append(pt)

# X座標（左右）でソートし、画面で一番左にある点を「親指」と仮定
fingertips.sort(key=lambda p: p[0])
thumb_tip = fingertips[0]

# 残りの指先候補から、Y座標（上下）でソートし、一番上にある点を「人差し指」と仮定
remaining_fingertips = fingertips[1:]
if len(remaining_fingertips) > 0:
    remaining_fingertips.sort(key=lambda p: p[1])
    index_tip = remaining_fingertips[0]
else:
    index_tip = fingertips[0]

C_distance_px = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
C_distance_cm = C_distance_px * cm_per_pixel

# 指先の点と距離を黄色で描画
cv2.circle(image_draw, tuple(thumb_tip), 8, (0, 255, 255), -1)
cv2.circle(image_draw, tuple(index_tip), 8, (0, 255, 255), -1)
cv2.line(image_draw, tuple(thumb_tip), tuple(index_tip), (0, 255, 255), 2)

# ======================================
# 6. 結果の出力と保存
# ======================================
print("\n========== 計測結果 ==========")
print(f"【指標A】 手の投影面積 : {A_area_cm2:.2f} cm^2")
print(f"【指標B】 手幅         : {B_width_cm:.2f} cm")
print(f"【指標C】 コの字の間隔 : {C_distance_cm:.2f} cm")
print("==============================\n")

# 画像にテキストで数値を書き込む
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image_draw, f"A(Area) : {A_area_cm2:.2f} cm2", (30, 50), font, 1, (0, 0, 255), 2)
cv2.putText(image_draw, f"B(Width): {B_width_cm:.2f} cm", (30, 90), font, 1, (255, 0, 0), 2)
cv2.putText(image_draw, f"C(C-Grip): {C_distance_cm:.2f} cm", (30, 130), font, 1, (0, 255, 255), 2)

# 画像を保存
result_path = os.path.join(RESULT_DIR, "result_ai.jpg")
cv2.imwrite(result_path, image_draw)
print(f"結果画像を保存しました: {result_path}")

# 画像を画面に表示 (何かキーを押すと閉じます)
cv2.imshow("Hand Measurement Result", image_draw)
cv2.waitKey(0)
cv2.destroyAllWindows()