import cv2
import mediapipe as mp
import json
import os

# ===============================
# 設定
# ===============================
VIDEO_PATH = "input.mp4"   # ← 自分の動画に変更
OUT_DIR = "demo_mp"        # 出力フォルダ

os.makedirs(OUT_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(VIDEO_PATH)

frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # 手が検出されたときだけJSON出力
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            points = []
            for lm in hand_landmarks.landmark:
                x = lm.x * w
                y = lm.y * h
                conf = 1.0   # MediaPipeはconfidenceが無いので固定
                points.extend([x, y, conf])

            data = {
                "people": [{
                    "hand_right_keypoints_2d": points
                }]
            }

            out_path = os.path.join(
                OUT_DIR,
                f"{frame_id:012d}_keypoints.json"
            )

            with open(out_path, "w") as f:
                json.dump(data, f)

    frame_id += 1

cap.release()
print("JSON作成 完了")