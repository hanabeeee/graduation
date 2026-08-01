import cv2
import os
import json
import mediapipe as mp

print(mp)
print(mp.__file__)

# ===== 設定 =====
VIDEO_PATH = r"//Users/hanabi/study/ゼミ/自学習/media/2026_2.mp4"  # ←動画パス
OUTPUT_DIR = r"/Users/hanabi/study/ゼミ/自学習/pipe/json.output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(VIDEO_PATH)

all_frames = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    print(f"Processing frame {frame_idx}")

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    landmarks_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                # x, y を保存（zは無視）
                landmarks_list.append([lm.x, lm.y])

    # 全フレーム保存（検出なしも含める）
    all_frames.append({
        "landmarks": landmarks_list
    })

    frame_idx += 1

cap.release()
print(f"Total valid frames: {len(all_frames)}")

# 動画ファイル名からJSON名を作成
video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
output_path = os.path.join(OUTPUT_DIR, f"{video_name}.json")

# まとめて保存
with open(output_path, "w") as f:
    json.dump(all_frames, f)

print(f"Saved to {output_path}")
print("JSON export completed")