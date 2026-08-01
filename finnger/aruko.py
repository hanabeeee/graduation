import cv2
import numpy as np
import os
import csv
import mediapipe as mp

# =======================================================
# 設定
# =======================================================
IMAGE_PATH = "finnger/media/hand3.png" # 撮影した画像のパス
RESULT_DIR = "result"
os.makedirs(RESULT_DIR, exist_ok=True)

# 定規で測った実際のマーカーの一辺の長さ(cm)
MARKER_REAL_SIZE_CM = 3.0 

# =======================================================
# 画像読込
# =======================================================
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"❌ 画像が見つかりません: {IMAGE_PATH}")
    exit()

result = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# =======================================================
# 1. ArUcoマーカーの検出とcm換算
# =======================================================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
corners, ids, rejected = detector.detectMarkers(gray)

CM_PER_PIXEL = None
marker_pixel_width = None

if ids is not None:
    cv2.aruco.drawDetectedMarkers(result, corners, ids)
    marker_corners = corners[0][0]
    marker_pixel_width = np.linalg.norm(marker_corners[0] - marker_corners[1])
    CM_PER_PIXEL = MARKER_REAL_SIZE_CM / marker_pixel_width
    print(f"🎉 ArUco検出成功! 1pixel = {CM_PER_PIXEL:.6f} cm")
else:
    print("❌ ArUcoマーカーが検出できません。")
    exit()

# =======================================================
# 2. MediaPipeによるAI骨格推定
# =======================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# MediaPipeの手検出モデルを初期化
with mp_hands.Hands(
    static_image_mode=True,       # 静止画モード
    max_num_hands=1,              # 検出する手の最大数
    min_detection_confidence=0.5  # 検出の厳しさ
) as hands:

    # MediaPipeはRGB画像を必要とするため、BGRから変換
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        print("❌ AIが手を検出できませんでした。")
        exit()

    # 検出された手のランドマーク（関節の座標）を取得
    hand_landmarks = results.multi_hand_landmarks[0]
    
    # 骨格を結果画像に描画（視覚的にわかりやすくする）
    mp_drawing.draw_landmarks(
        result, 
        hand_landmarks, 
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
    )

    # =======================================================
    # 3. 特徴点抽出とA・B・C計測（画像に対して完全な垂直・水平版）
    # =======================================================
    h, w, _ = image.shape

    # AIが出力した相対座標をピクセル座標に変換する関数
    def get_pt(landmark_id):
        lm = hand_landmarks.landmark[landmark_id]
        return np.array([int(lm.x * w), int(lm.y * h)])

    # 💡 解剖学的な特徴点を取得
    wrist = get_pt(mp_hands.HandLandmark.WRIST)                   # 0: 手首
    middle_mcp = get_pt(mp_hands.HandLandmark.MIDDLE_FINGER_MCP)  # 9: 中指の付け根
    middle_tip = get_pt(mp_hands.HandLandmark.MIDDLE_FINGER_TIP)  # 12: 中指の先端
    thumb_mcp = get_pt(mp_hands.HandLandmark.THUMB_MCP)           # 2: 親指の付け根
    pinky_mcp = get_pt(mp_hands.HandLandmark.PINKY_MCP)           # 17: 小指の付け根

    # 🔥 【修正】斜めを許容せず、画像の「縦(Y)・横(X)」軸に沿った純粋な直線距離を測る
    # A: 中指の先端(Y) 〜 中指の付け根(Y) の純粋な縦の長さ
    A_px = abs(middle_mcp[1] - middle_tip[1])
    
    # B: 中指の付け根(Y) 〜 手首(Y) の純粋な縦の長さ
    B_px = abs(wrist[1] - middle_mcp[1])
    
    # C: 手の横幅（親指の付け根(X) 〜 小指の付け根(X) の純粋な横幅）
    C_px = abs(thumb_mcp[0] - pinky_mcp[0])

    # --- 物理単位変換（cm） ---
    A_cm = A_px * CM_PER_PIXEL
    B_cm = B_px * CM_PER_PIXEL
    C_cm = C_px * CM_PER_PIXEL

    # --- 描画用の座標計算（基準画像と同じように垂直・水平な直線を引く） ---
    # Aの矢印（中指の付け根のX座標を基準に、真上へ引く）
    a_start = (middle_mcp[0], middle_mcp[1])
    a_end = (middle_mcp[0], middle_tip[1])

    # Bの矢印（中指の付け根のX座標を基準に、真下へ引く）
    b_start = (middle_mcp[0], middle_mcp[1])
    b_end = (middle_mcp[0], wrist[1])

    # Cの矢印（見栄えを良くするため、Y座標は親指と小指の中間あたりに設定して真横へ引く）
    c_y = int((thumb_mcp[1] + pinky_mcp[1]) / 2)
    c_start = (thumb_mcp[0], c_y)
    c_end = (pinky_mcp[0], c_y)

    # --- 結果の矢印とテキストを描画 ---
    cv2.arrowedLine(result, a_start, a_end, (255, 0, 255), 2, tipLength=0.05)
    cv2.arrowedLine(result, b_start, b_end, (255, 255, 0), 2, tipLength=0.05)
    cv2.arrowedLine(result, c_start, c_end, (0, 255, 255), 2, tipLength=0.05)

    cv2.putText(result, f"A: {A_cm:.2f}cm", (a_end[0] - 80, a_end[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(result, f"B: {B_cm:.2f}cm", (b_start[0] + 15, b_start[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(result, f"C: {C_cm:.2f}cm", (c_start[0] + 20, c_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    print("\n===== 🤖 AI計測結果（垂直・水平版） =====")
    print(f"A (指の長さ)   : {A_cm:.2f} cm")
    print(f"B (手のひら長) : {B_cm:.2f} cm")
    print(f"C (手の横幅)   : {C_cm:.2f} cm")
    print("==================================\n")

# =======================================================
# 4. CSV ＆ 結果画像の保存
# =======================================================
csv_path = os.path.join(RESULT_DIR, "result.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "MarkerSize(px)", "CM_PER_PIXEL", "A(cm)", "B(cm)", "C(cm)"])
    if results.multi_hand_landmarks:
        writer.writerow([
            os.path.basename(IMAGE_PATH),
            round(marker_pixel_width, 2),
            round(CM_PER_PIXEL, 6),
            round(A_cm, 2),
            round(B_cm, 2),
            round(C_cm, 2)
        ])

cv2.imwrite(os.path.join(RESULT_DIR, "result_ai.jpg"), result)
print("処理完了。骨格が描画された result_ai.jpg を確認してください。")