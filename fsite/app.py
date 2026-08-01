import cv2
import numpy as np
import base64
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ======================================
# 設定
# ======================================
# ArUcoマーカーの1辺の長さ(cm)
MARKER_REAL_SIZE_CM = 3.0

# MediaPipeの手検出モデルを初期化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({"error": "画像ファイルが送信されていません"}), 400

    file = request.files['file']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "画像の読み込みに失敗しました"}), 400

    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w, _ = image.shape

    # =======================================================
    # 1. ArUcoマーカーの検出とcm換算
    # =======================================================
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    
    try:
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(image)
    except AttributeError:
        # OpenCVの古いバージョン対応
        corners, ids, rejected = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)

    CM_PER_PIXEL = None

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(result, corners, ids)
        marker_corners = corners[0][0]
        marker_pixel_width = np.linalg.norm(marker_corners[0] - marker_corners[1])
        CM_PER_PIXEL = MARKER_REAL_SIZE_CM / marker_pixel_width
    else:
        return jsonify({"error": "ArUcoマーカーが検出できませんでした"}), 400

    # =======================================================
    # 2. MediaPipeによるAI骨格推定と計測
    # =======================================================
    # MediaPipeはRGB画像を必要とするため変換
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_hands.Hands(
        static_image_mode=True,       
        max_num_hands=1,              
        min_detection_confidence=0.5  
    ) as hands:
        
        ai_results = hands.process(image_rgb)

        if not ai_results.multi_hand_landmarks:
            return jsonify({"error": "AIが手を検出できませんでした"}), 400

        # 検出された手のランドマークを取得
        hand_landmarks = ai_results.multi_hand_landmarks[0]
        
        # 骨格を結果画像に描画
        mp_drawing.draw_landmarks(
            result, 
            hand_landmarks, 
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
        )

        # AIが出力した相対座標をピクセル座標に変換する関数
        def get_pt(landmark_id):
            lm = hand_landmarks.landmark[landmark_id]
            return np.array([int(lm.x * w), int(lm.y * h)])

        # 特徴点の取得
        wrist = get_pt(mp_hands.HandLandmark.WRIST)                   # 0: 手首
        middle_mcp = get_pt(mp_hands.HandLandmark.MIDDLE_FINGER_MCP)  # 9: 中指の付け根
        middle_tip = get_pt(mp_hands.HandLandmark.MIDDLE_FINGER_TIP)  # 12: 中指の先端
        thumb_mcp = get_pt(mp_hands.HandLandmark.THUMB_MCP)           # 2: 親指の付け根
        pinky_mcp = get_pt(mp_hands.HandLandmark.PINKY_MCP)           # 17: 小指の付け根

        # --- 純粋な縦横の直線距離(px) ---
        A_px = abs(middle_mcp[1] - middle_tip[1])
        B_px = abs(wrist[1] - middle_mcp[1])
        C_px = abs(thumb_mcp[0] - pinky_mcp[0])

        # --- 物理単位変換（cm） ---
        A_cm = A_px * CM_PER_PIXEL
        B_cm = B_px * CM_PER_PIXEL
        C_cm = C_px * CM_PER_PIXEL

        # --- 描画用の座標計算（垂直・水平な直線を引く） ---
        a_start = (middle_mcp[0], middle_mcp[1])
        a_end = (middle_mcp[0], middle_tip[1])

        b_start = (middle_mcp[0], middle_mcp[1])
        b_end = (middle_mcp[0], wrist[1])

        c_y = int((thumb_mcp[1] + pinky_mcp[1]) / 2)
        c_start = (thumb_mcp[0], c_y)
        c_end = (pinky_mcp[0], c_y)

        # 矢印とテキストを描画
        cv2.arrowedLine(result, a_start, a_end, (255, 0, 255), 2, tipLength=0.05)
        cv2.arrowedLine(result, b_start, b_end, (255, 255, 0), 2, tipLength=0.05)
        cv2.arrowedLine(result, c_start, c_end, (0, 255, 255), 2, tipLength=0.05)

        cv2.putText(result, f"A: {A_cm:.2f}cm", (a_end[0] - 80, a_end[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(result, f"B: {B_cm:.2f}cm", (b_start[0] + 15, b_start[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(result, f"C: {C_cm:.2f}cm", (c_start[0] + 20, c_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 3. 解析済み画像をWebで表示できるようにBase64に変換
    _, buffer = cv2.imencode('.jpg', result)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # 4. JSON形式でフロントエンド（Webサイト）に返す（float型のエラー対策済み）
    return jsonify({
        "status": "success",
        "A_area_cm2": round(float(A_cm), 2),  # 指標A（指の長さ）
        "B_width_cm": round(float(B_cm), 2),  # 指標B（手のひらの長さ）
        "C_grip_cm": round(float(C_cm), 2),   # 指標C（手の横幅）
        "processed_image_base64": f"data:image/jpeg;base64,{img_base64}"
    })

if __name__ == '__main__':
    print("サーバーを起動しました！ http://127.0.0.1:5001/analyze で待機中...")
    app.run(host='0.0.0.0', port=5001, debug=True)