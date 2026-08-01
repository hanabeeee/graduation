import cv2

# 動画を読み込み
cap = cv2.VideoCapture('hokou1.mp4')

# 保存したいフレーム番号（0が最初）
frame_no = 25  # たとえば1秒目(30fpsなら)

# フレーム位置を指定
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

# フレームを取得
ret, frame = cap.read()

# 取得できたら保存
if ret:
    cv2.imwrite('frame_25.png', frame)
    print("frame_25.png として保存しました！")
else:
    print("フレームを取得できませんでした。")

cap.release()
