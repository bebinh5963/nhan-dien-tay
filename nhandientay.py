import cv2
import mediapipe as mp

# 1. Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 2. Mở webcam (thường là số 0)
cap = cv2.VideoCapture(0)

print("Dang mo Webcam... Nhan phim 'ESC' de thoat.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Khong the doc du lieu tu Webcam.")
        break

    # Lật ảnh để tạo hiệu ứng soi gương (Mirror effect)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Chuyển ảnh từ BGR sang RGB cho MediaPipe xử lý
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # 3. Nếu phát hiện ra người trong khung hình
    if results.pose_landmarks:
        # Vẽ khung xương để dễ nhìn
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        landmarks = results.pose_landmarks.landmark

        # Lấy tọa độ Y của Mũi làm mốc (càng lên cao Y càng nhỏ)
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
        
        # --- LẤY TỌA ĐỘ TỪ MEDIAPIPE ---
        # Điểm LEFT_WRIST của MediaPipe (trên ảnh đã lật)
        mp_left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        left_wrist_x = int(mp_left_wrist.x * w)
        left_wrist_y = mp_left_wrist.y
        left_wrist_y_px = int(left_wrist_y * h)

        # Điểm RIGHT_WRIST của MediaPipe (trên ảnh đã lật)
        mp_right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        right_wrist_x = int(mp_right_wrist.x * w)
        right_wrist_y = mp_right_wrist.y
        right_wrist_y_px = int(right_wrist_y * h)


        # --- 4. LOGIC ĐÃ SỬA LỖI NGƯỢC GƯƠNG ---
        
        # Do ảnh bị lật ngược (flip), tay trái của MediaPipe giờ là TAY PHẢI thật của bạn
        if left_wrist_y < nose_y:
            cv2.putText(frame, "TAY PHAI (Right)", (left_wrist_x - 60, left_wrist_y_px - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.circle(frame, (left_wrist_x, left_wrist_y_px), 12, (0, 0, 255), -1)

        # Và ngược lại, tay phải của MediaPipe giờ là TAY TRÁI thật của bạn
        if right_wrist_y < nose_y:
            cv2.putText(frame, "TAY TRAI (Left)", (right_wrist_x - 60, right_wrist_y_px - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.circle(frame, (right_wrist_x, right_wrist_y_px), 12, (0, 255, 0), -1)

    # 5. Hiển thị cửa sổ
    cv2.imshow("Nhan dien Tay Trai/Phai", frame)

    # Nhấn phím ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 6. Dọn dẹp tài nguyên sau khi tắt
cap.release()
cv2.destroyAllWindows()
pose.close()