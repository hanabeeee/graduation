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
arm_raise_count = 0
arm_up = False
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
            pose_detected = False
            hand_count = 0
            face_count = 0
            avg_pose_score = 0.0
            posture_text = "No Pose"
            shoulder_status = "Unknown"
            neck_status = "Unknown"
            posture_score = 0
            arm_angle = 0

            # ==========================
            # Pose（33点）
            # ==========================
            if pose_results.pose_landmarks:
                pose_detected = True
                avg_pose_score = sum(lm.visibility for lm in pose_results.pose_landmarks.landmark) / len(pose_results.pose_landmarks.landmark)

                left_shoulder = pose_results.pose_landmarks.landmark[11]
                right_shoulder = pose_results.pose_landmarks.landmark[12]
                left_ear = pose_results.pose_landmarks.landmark[7]
                right_ear = pose_results.pose_landmarks.landmark[8]

                shoulder_diff = abs(left_shoulder.y - right_shoulder.y)

                if shoulder_diff < 0.03:
                    shoulder_status = "Good"
                elif shoulder_diff < 0.07:
                    shoulder_status = "Slight Tilt"
                else:
                    shoulder_status = "Bad"

                shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
                ear_center_x = (left_ear.x + right_ear.x) / 2
                neck_forward = abs(ear_center_x - shoulder_center_x)

                if neck_forward < 0.03:
                    neck_status = "Good"
                elif neck_forward < 0.07:
                    neck_status = "Forward Head"
                else:
                    neck_status = "Severe Forward Head"

                posture_score = 100
                posture_score -= min(int(shoulder_diff * 500), 40)
                posture_score -= min(int(neck_forward * 500), 40)
                posture_score = max(0, min(100, posture_score))

                if posture_score >= 80:
                    posture_text = "A"
                elif posture_score >= 60:
                    posture_text = "B"
                else:
                    posture_text = "C"

                left_shoulder = pose_results.pose_landmarks.landmark[11]
                left_elbow = pose_results.pose_landmarks.landmark[13]
                left_wrist = pose_results.pose_landmarks.landmark[15]

                sx, sy = left_shoulder.x, left_shoulder.y
                ex, ey = left_elbow.x, left_elbow.y
                wx, wy = left_wrist.x, left_wrist.y

                import math

                v1 = (sx - ex, sy - ey)
                v2 = (wx - ex, wy - ey)

                dot = v1[0] * v2[0] + v1[1] * v2[1]
                mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
                mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

                if mag1 > 0 and mag2 > 0:
                    cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                    arm_angle = math.degrees(math.acos(cos_theta))

                if left_wrist.y < left_shoulder.y and not arm_up:
                    arm_up = True

                if left_wrist.y > left_shoulder.y and arm_up:
                    arm_raise_count += 1
                    arm_up = False

                mp_draw.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                for idx, lm in enumerate(
                        pose_results.pose_landmarks.landmark):

                    score = lm.visibility
                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                    cv2.putText(
                        frame,
                        f"{idx}:{score:.2f}",
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
                hand_count = len(hand_results.multi_hand_landmarks)

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
                face_count = len(face_results.multi_face_landmarks)

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

            cv2.putText(frame, f"Pose: {pose_detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"PoseScore: {avg_pose_score:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Hands: {hand_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {face_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Posture Rank: {posture_text}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Shoulder: {shoulder_status}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Neck: {neck_status}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Score: {posture_score}/100", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Arm Angle: {arm_angle:.0f}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Arm Raise Count: {arm_raise_count}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("MediaPipe Pose + Hands + Face", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\nCtrl+C detected. Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()