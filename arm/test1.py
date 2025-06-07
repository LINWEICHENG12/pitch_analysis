import cv2
import mediapipe as mp
import time
import math
from collections import deque
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

class ArmAxisTracker:
    def __init__(self, max_history=300, pixels_per_meter=300):
        self.max_history = max_history
        self.pixels_per_meter = pixels_per_meter
        self.axis_trajectory = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
        self.total_displacement = 0.0
        self.max_speed = 0.0
        self.speeds = deque(maxlen=50)
        self.angles = deque(maxlen=max_history)
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue(maxsize=30)
        self.running = True
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.speed_history = []
        self.time_history = []
        self.recording = False
        self.record_start_time = 0

    def add_axis_position(self, shoulder, elbow, wrist, timestamp):
        """添加肩膀、手肘和手腕位置，計算速度和角度"""
        if not shoulder or not elbow or not wrist:
            return

        # 計算手肘位置
        self.axis_trajectory.append(elbow)
        self.timestamps.append(timestamp)

        # 計算角度
        angle = self._calculate_angle(shoulder, elbow, wrist)
        self.angles.append(angle)

        # 計算速度
        if len(self.timestamps) >= 2:
            self._calculate_speed()
            
            # 如果正在記錄，保存速度和時間
            if self.recording:
                self.speed_history.append(self.speeds[-1])
                self.time_history.append(timestamp - self.record_start_time)

    def _calculate_angle(self, shoulder, elbow, wrist):
        """計算肩膀、手肘和手腕的夾角"""
        # 向量1：肩膀 -> 手肘
        v1 = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
        # 向量2：手腕 -> 手肘
        v2 = (wrist[0] - elbow[0], wrist[1] - elbow[1])

        # 計算向量的內積和模長
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return 0.0

        # 計算角度（弧度轉角度）
        angle = math.acos(dot_product / (magnitude_v1 * magnitude_v2))
        return math.degrees(angle)

    def _calculate_speed(self):
        """計算手肘速度"""
        curr = self.axis_trajectory[-1]
        prev = self.axis_trajectory[-2]
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        dist_px = math.sqrt(dx**2 + dy**2)

        dt = self.timestamps[-1] - self.timestamps[-2]
        if dt <= 0:
            return

        # 計算速度 (將時間從毫秒轉換為秒)
        dist_m = dist_px / self.pixels_per_meter
        speed = (dist_m / dt) * 1000  # 將毫秒轉換為秒
        self.speeds.append(speed)

        # 更新最大速度
        if speed > self.max_speed:
            self.max_speed = speed

    def draw_trajectory(self, image):
        """在影像上繪製手軸軌跡（連續線條）"""
        if len(self.axis_trajectory) < 2:
            return image

        # 使用 numpy 進行向量化運算
        points = np.array(self.axis_trajectory, dtype=np.int32)
        cv2.polylines(image, [points], False, (0, 255, 0), 2)
        return image

    def draw_statistics(self, image):
        """在影像上顯示統計數據"""
        # 計算 FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            self.fps = self.frame_count / elapsed_time

        cv2.putText(image, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"Max Speed: {self.max_speed:.2f} m/s",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"Total Distance: {self.total_displacement:.2f} m",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if self.angles:
            cv2.putText(image, f"Current Angle: {self.angles[-1]:.2f} degrees",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return image

    def start_recording(self):
        """開始記錄速度數據"""
        self.recording = True
        self.speed_history = []
        self.time_history = []
        self.record_start_time = time.time()
        print("📝 開始記錄速度數據")

    def stop_recording(self):
        """停止記錄速度數據"""
        self.recording = False
        print("⏹ 停止記錄速度數據")

    def export_speed_graph(self):
        """匯出速度-時間圖表"""
        if not self.speed_history or not self.time_history:
            print("❌ 沒有可匯出的數據")
            return

        # 創建圖表
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_history, self.speed_history, 'b-', label='手軸移動速度')
        plt.title('手軸移動速度隨時間變化')
        plt.xlabel('時間 (秒)')
        plt.ylabel('速度 (m/s)')
        plt.grid(True)
        plt.legend()

        # 創建輸出目錄
        output_dir = 'speed_graphs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 生成檔案名稱（使用時間戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'speed_graph_{timestamp}.png'
        filepath = os.path.join(output_dir, filename)

        # 保存圖表
        plt.savefig(filepath)
        plt.close()
        print(f"✅ 圖表已保存至: {filepath}")

    def process_frame(self, frame, pose):
        """處理單一幀影像"""
        if frame is None:
            return None

        current_time = time.time()
        frame = cv2.flip(frame, 1)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_pose = mp.solutions.pose
            mp_drawing.draw_landmarks(
                image_bgr,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
            )

            h, w, _ = image_bgr.shape
            lm = results.pose_landmarks.landmark
            shoulder = (
                int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)
            )
            elbow = (
                int(lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w),
                int(lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h)
            )
            wrist = (
                int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * w),
                int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * h)
            )

            self.add_axis_position(shoulder, elbow, wrist, current_time)
            image_bgr = self.draw_trajectory(image_bgr)
            image_bgr = self.draw_statistics(image_bgr)

        return image_bgr

def main():
    """主函數 - 手軸追蹤"""
    print("🚀 手軸追蹤系統啟動")
    print("=" * 50)
    
    try:
        print(f"📋 OpenCV 版本: {cv2.__version__}")
        print(f"🧠 MediaPipe 版本: {mp.__version__}")
    except Exception as e:
        print(f"⚠️ 無法獲取版本信息: {e}")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    arm_tracker = ArmAxisTracker()

    print("🎥 正在初始化攝影機...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法開啟攝影機")
        print("建議檢查:")
        print("1. 攝影機是否被其他程式使用")
        print("2. 攝影機驅動程式是否正常")
        print("3. Windows 隱私設定是否允許攝影機存取")
        return
    
    print("✅ 攝影機初始化成功")
    
    # 設置攝影機參數
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        print("🤖 正在初始化 MediaPipe Pose...")
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            print("✅ MediaPipe Pose 初始化成功")
            print("📝 控制說明:")
            print("p: 暫停/恢復")
            print("r: 重置")
            print("s: 開始/停止記錄")
            print("e: 匯出圖表")
            print("ESC: 退出")
            
            paused = False
            recording = False

            def capture_frames():
                print("📸 開始擷取影像...")
                frame_count = 0
                while arm_tracker.running:
                    if not paused:
                        ret, frame = cap.read()
                        if ret:
                            arm_tracker.frame_queue.put(frame)
                            frame_count += 1
                            if frame_count % 30 == 0:  # 每30幀顯示一次狀態
                                print(f"📊 已擷取 {frame_count} 幀影像")
                        else:
                            print("❌ 無法讀取影像，跳出迴圈")
                            break
                    time.sleep(0.001)
                print("📸 停止擷取影像")

            def process_frames():
                print("🔄 開始處理影像...")
                frame_count = 0
                while arm_tracker.running:
                    if not paused and not arm_tracker.frame_queue.empty():
                        try:
                            frame = arm_tracker.frame_queue.get(timeout=1.0)
                            processed_frame = arm_tracker.process_frame(frame, pose)
                            if processed_frame is not None:
                                arm_tracker.result_queue.put(processed_frame)
                                frame_count += 1
                                if frame_count % 30 == 0:  # 每30幀顯示一次狀態
                                    print(f"🔄 已處理 {frame_count} 幀影像")
                        except queue.Empty:
                            print("⚠️ 等待影像中...")
                            continue
                    time.sleep(0.001)
                print("🔄 停止處理影像")

            # 創建執行緒池
            print("🧵 啟動執行緒...")
            with ThreadPoolExecutor(max_workers=2) as executor:
                # 啟動執行緒
                capture_thread = executor.submit(capture_frames)
                process_thread = executor.submit(process_frames)

                print("✅ 系統初始化完成，開始追蹤...")
                while True:
                    if not arm_tracker.result_queue.empty():
                        try:
                            frame = arm_tracker.result_queue.get(timeout=1.0)
                            cv2.putText(frame, "p:Pause/Resume  r:Reset  s:Record  e:Export  ESC:Exit",
                                        (10, frame.shape[0] - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.imshow('Arm Axis Tracker', frame)
                        except queue.Empty:
                            print("⚠️ 等待處理結果中...")
                            continue

                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        print("👋 使用者要求退出")
                        break
                    elif key == ord('r'):  # Reset
                        arm_tracker = ArmAxisTracker()
                        print("🔄 軌跡已重置")
                    elif key == ord('p'):  # Pause/Resume
                        paused = not paused
                        print("⏸ 暫停" if paused else "▶️ 恢復")
                    elif key == ord('s'):  # Start/Stop Recording
                        if not arm_tracker.recording:
                            arm_tracker.start_recording()
                        else:
                            arm_tracker.stop_recording()
                    elif key == ord('e'):  # Export Graph
                        arm_tracker.export_speed_graph()

                # 停止執行緒
                print("🛑 正在停止執行緒...")
                arm_tracker.running = False
                capture_thread.result()
                process_thread.result()
                print("✅ 執行緒已停止")

    except Exception as e:
        print(f"❌ 主程式崩潰: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🧹 正在釋放資源...")
        cap.release()
        cv2.destroyAllWindows()
        print("✅ 資源已釋放")

if __name__ == "__main__":
    main()