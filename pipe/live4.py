import cv2
import mediapipe as mp

# ==========================
# MediaPipe 初期設定
# ==========================
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

# 顔の主要ランドマーク
FACE_POINTS = [1, 33, 263, 61, 291, 152]

# ==========================
# カメラ起動
# ==========================
cap = cv2.VideoCapture(0)
try:
        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_results = pose.process(rgb)
            hand_results = hands.process(rgb)
            face_results = face.process(rgb)

            h, w, _ = frame.shape

            # ==========================
            # Pose（33点）
            # ==========================
            if pose_results.pose_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                for idx, lm in enumerate(
                        pose_results.pose_landmarks.landmark):

                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                    cv2.putText(
                        frame,
                        str(idx),
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 0),
                        1
                    )

            # ==========================
            # Hands（21点）
            # ==========================
            if hand_results.multi_hand_landmarks:

                for hand_landmarks in hand_results.multi_hand_landmarks:

                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    for idx, lm in enumerate(hand_landmarks.landmark):

                        x = int(lm.x * w)
                        y = int(lm.y * h)

                        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

                        cv2.putText(
                            frame,
                            str(idx),
                            (x + 3, y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (0, 0, 255),
                            1
                        )

            # ==========================
            # Face（主要6点のみ）
            # ==========================
            if face_results.multi_face_landmarks:

                for face_landmarks in face_results.multi_face_landmarks:

                    for idx in FACE_POINTS:

                        lm = face_landmarks.landmark[idx]

                        x = int(lm.x * w)
                        y = int(lm.y * h)

                        cv2.circle(frame, (x, y), 6, (255, 255, 0), -1)

                        cv2.putText(
                            frame,
                            str(idx),
                            (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1
                        )

            cv2.imshow("MediaPipe Pose + Hands + Face", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\nCtrl+C detected. Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()