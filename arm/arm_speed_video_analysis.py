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
import sys

class ArmAxisTracker:
    def __init__(self, max_history=300, pixels_per_meter=346.8):
        self.max_history = max_history
        self.pixels_per_meter = pixels_per_meter
        self.positions = deque(maxlen=max_history)
        self.speeds = deque(maxlen=max_history)
        self.angles = deque(maxlen=max_history)
        self.time_history = deque(maxlen=max_history)
        self.speed_history = deque(maxlen=max_history)
        self.wrist_speed_history = deque(maxlen=max_history)  # 新增：手掌速度歷史
        self.elbow_speed_history = deque(maxlen=max_history)  # 新增：手肘速度歷史
        self.wrist_acceleration_history = deque(maxlen=max_history)  # 新增：手掌加速度歷史
        self.recording = False
        self.record_start_time = 0
        self.video_path = None
        self.video_fps = 0
        self.video_frame_count = 0
        self.current_frame = 0
        self.playback_speed = 1.0  # 播放速度倍率
        self.video_width = 0
        self.video_height = 0
        self.cap = None
        
        # 初始化 MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def load_video(self, video_path):
        """載入影片"""
        try:
            # 處理路徑
            video_path = os.path.abspath(video_path)
            print(f"📂 正在嘗試開啟影片: {video_path}")
            print(f"📂 當前工作目錄: {os.getcwd()}")
            
            if not os.path.exists(video_path):
                print(f"❌ 錯誤：找不到影片檔案 '{video_path}'")
                print("請確認：")
                print("1. 檔案路徑是否正確")
                print("2. 檔案名稱是否正確（包括副檔名）")
                print("3. 檔案是否在正確的目錄中")
                return False

            # 檢查檔案權限
            if not os.access(video_path, os.R_OK):
                print(f"❌ 錯誤：無法讀取檔案 '{video_path}'")
                print("請確認檔案權限是否正確")
                return False

            self.video_path = video_path
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                print("❌ 錯誤：無法開啟影片")
                self.cap.release()  # 確保釋放資源
                print("🔄 已釋放部分初始化的資源")
                print("可能的原因：")
                print("1. 檔案格式不支援")
                print("2. 檔案損壞")
                print("3. 檔案路徑包含特殊字元")
                print("4. 檔案權限問題")
                return False
                
            # 獲取影片資訊
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if self.video_fps <= 0 or self.video_frame_count <= 0:
                print("❌ 錯誤：無法讀取影片資訊")
                print(f"FPS: {self.video_fps}")
                print(f"總幀數: {self.video_frame_count}")
                return False
            
            print(f"📹 影片資訊:")
            print(f"解析度: {self.video_width}x{self.video_height}")
            print(f"幀率: {self.video_fps} FPS")
            print(f"總幀數: {self.video_frame_count}")
            print(f"檔案大小: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
            
            # 測試讀取第一幀
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("❌ 錯誤：無法讀取影片幀")
                return False
                
            # 重置影片位置
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            return True
            
        except Exception as e:
            print(f"❌ 載入影片時發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def add_axis_position(self, shoulder, elbow, wrist, timestamp):
        """添加手軸位置"""
        if len(self.positions) > 0:
            prev_shoulder, prev_elbow, prev_wrist = self.positions[-1]
            
            # 計算手掌速度
            dx_wrist = abs(wrist[0] - prev_wrist[0])
            dy_wrist = abs(wrist[1] - prev_wrist[1])
            distance_wrist = math.sqrt(dx_wrist**2 + dy_wrist**2)
            
            # 計算手肘速度
            dx_elbow = abs(elbow[0] - prev_elbow[0])
            dy_elbow = abs(elbow[1] - prev_elbow[1])
            distance_elbow = math.sqrt(dx_elbow**2 + dy_elbow**2)
            
            time_diff = timestamp - self.time_history[-1]
            if time_diff > 0:
                # 計算手掌速度
                distance_wrist_m = distance_wrist / self.pixels_per_meter
                wrist_speed = distance_wrist_m / time_diff
                self.wrist_speed_history.append(wrist_speed)
                
                # 計算手肘速度
                distance_elbow_m = distance_elbow / self.pixels_per_meter
                elbow_speed = distance_elbow_m / time_diff
                self.elbow_speed_history.append(elbow_speed)
            else:
                self.wrist_speed_history.append(0)
                self.elbow_speed_history.append(0)
        else:
            self.wrist_speed_history.append(0)
            self.elbow_speed_history.append(0)

        # 計算角度
        angle = self.calculate_angle(shoulder, elbow, wrist)
        self.angles.append(angle)
        
        self.positions.append((shoulder, elbow, wrist))
        self.time_history.append(timestamp)

    def calculate_angle(self, shoulder, elbow, wrist):
        """計算手軸角度"""
        # 計算向量
        v1 = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
        v2 = (wrist[0] - elbow[0], wrist[1] - elbow[1])
        
        # 計算角度
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        v1_mag = math.sqrt(v1[0]**2 + v1[1]**2)
        v2_mag = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if v1_mag * v2_mag == 0:
            return 0
            
        cos_angle = dot_product / (v1_mag * v2_mag)
        cos_angle = max(-1, min(1, cos_angle))  # 確保在 [-1, 1] 範圍內
        angle = math.degrees(math.acos(cos_angle))
        
        return angle

    def draw_trajectory(self, image):
        """繪製軌跡（只保留手掌）"""
        if len(self.positions) < 2:
            return image
        # 只繪製手掌（wrist）軌跡線
        for i in range(1, len(self.positions)):
            prev_shoulder, prev_elbow, prev_wrist = self.positions[i-1]
            curr_shoulder, curr_elbow, curr_wrist = self.positions[i]
            alpha = i / len(self.positions)
            color = (0, int(255 * (1-alpha)), int(255 * alpha))
            cv2.line(image, prev_wrist, curr_wrist, color, 2)
        return image

    def draw_statistics(self, image):
        """繪製統計資訊 (已移除即時顯示功能)"""
        return image  # 保留空方法以防止其他地方調用出錯

    def process_video_frame(self, frame):
        """處理影片幀 (僅保留軌跡描繪功能)"""
        if frame is None:
            return None

        try:
            # 轉換顏色空間
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 繪製骨架
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # 獲取關鍵點
                landmarks = results.pose_landmarks.landmark
                shoulder = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * self.video_width),
                           int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * self.video_height))
                elbow = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * self.video_width),
                        int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * self.video_height))
                wrist = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * self.video_width),
                        int(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * self.video_height))

                # 添加位置和計算速度
                current_time = self.current_frame / self.video_fps
                self.add_axis_position(shoulder, elbow, wrist, current_time)

                # 繪製軌跡
                image = self.draw_trajectory(image)

            return image

        except Exception as e:
            print(f"❌ 處理影片幀時發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame

    def start_recording(self):
        """開始記錄"""
        self.recording = True
        self.record_start_time = time.time()
        print("🎥 開始記錄")

    def stop_recording(self):
        """停止記錄"""
        self.recording = False
        print("⏹ 停止記錄")

    def export_analysis_report(self):
        """Export analysis report"""
        if not self.wrist_speed_history or not self.elbow_speed_history or not self.time_history:
            print("❌ No data to export")
            return

        # Create output directory
        output_dir = 'analysis_reports'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f'analysis_report_{timestamp}'

        # Calculate statistics
        max_wrist_speed = max(self.wrist_speed_history)
        max_elbow_speed = max(self.elbow_speed_history)
        avg_wrist_speed = sum(self.wrist_speed_history)/len(self.wrist_speed_history)
        avg_elbow_speed = sum(self.elbow_speed_history)/len(self.elbow_speed_history)
        max_angle = max(self.angles) if self.angles else 0
        min_angle = min(self.angles) if self.angles else 0

        # Speed-time chart
        plt.figure(figsize=(12, 6))
        plt.plot(self.time_history, self.wrist_speed_history, 'g-', label='Wrist Speed')
        plt.plot(self.time_history, self.elbow_speed_history, 'r-', label='Elbow Speed')
        plt.title(f'Arm Movement Speed Analysis\n'
                 f'Max Wrist Speed: {max_wrist_speed:.2f} m/s | Max Elbow Speed: {max_elbow_speed:.2f} m/s\n'
                 f'Video Info: {self.video_width}x{self.video_height} @ {self.video_fps}fps',
                 fontsize=12, pad=20)
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('Speed (m/s)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.text(0.02, 0.98, 
                f'Max Wrist Speed: {max_wrist_speed:.2f} m/s\n'
                f'Avg Wrist Speed: {avg_wrist_speed:.2f} m/s\n'
                f'Max Elbow Speed: {max_elbow_speed:.2f} m/s\n'
                f'Avg Elbow Speed: {avg_elbow_speed:.2f} m/s',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{base_filename}_speed.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 加速度圖表
        wrist_acceleration = [0]
        for i in range(1, len(self.wrist_speed_history)):
            dt = self.time_history[i] - self.time_history[i-1]
            if dt > 1e-6:
                acc = (self.wrist_speed_history[i] - self.wrist_speed_history[i-1]) / dt
            else:
                acc = 0
            wrist_acceleration.append(acc)
        plt.figure(figsize=(12, 6))
        plt.plot(self.time_history, wrist_acceleration, 'b-', label='Wrist Acceleration')
        plt.title(f'Wrist Acceleration Analysis\n'
                 f'Video Info: {self.video_width}x{self.video_height} @ {self.video_fps}fps',
                 fontsize=12, pad=20)
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('Acceleration (m/s²)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{base_filename}_acceleration.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Generate text report
        report_path = os.path.join(output_dir, f'{base_filename}_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Arm Movement Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Video Information:\n")
            f.write(f"Resolution: {self.video_width}x{self.video_height}\n")
            f.write(f"Frame Rate: {self.video_fps} FPS\n")
            f.write(f"Total Frames: {self.video_frame_count}\n")
            f.write(f"File Size: {os.path.getsize(self.video_path) / (1024*1024):.2f} MB\n\n")
            f.write("Movement Data:\n")
            f.write(f"Max Wrist Speed: {max_wrist_speed:.2f} m/s\n")
            f.write(f"Avg Wrist Speed: {avg_wrist_speed:.2f} m/s\n")
            f.write(f"Max Elbow Speed: {max_elbow_speed:.2f} m/s\n")
            f.write(f"Avg Elbow Speed: {avg_elbow_speed:.2f} m/s\n")
            if self.angles:
                f.write(f"Max Angle: {max_angle:.1f} degrees\n")
                f.write(f"Min Angle: {min_angle:.1f} degrees\n")
            f.write("\nTime Point Analysis:\n")
            for i, (t, ws, es, acc) in enumerate(zip(self.time_history, self.wrist_speed_history, self.elbow_speed_history, wrist_acceleration)):
                f.write(f"Time {t:.2f}s: Wrist Speed {ws:.2f} m/s, Elbow Speed {es:.2f} m/s, Wrist Acc {acc:.2f} m/s²")
                if self.angles:
                    f.write(f", Angle {self.angles[i]:.1f} degrees")
                f.write("\n")
        print(f"✅ Analysis report saved to: {output_dir}")
        print(f"📊 Generated charts:")
        print(f"   - {base_filename}_speed.png")
        print(f"   - {base_filename}_acceleration.png")
        print(f"📝 Generated report: {base_filename}_report.txt")

    def release_resources(self):
        """釋放資源"""
        if self.cap is not None:
            self.cap.release()
            print("✅ ArmAxisTracker 資源已釋放")

def main():
    """主函數 - 手軸追蹤"""
    print("\n" + "=" * 50)
    print("🚀 手軸追蹤系統啟動")
    print("=" * 50)
    
    # 顯示系統資訊
    print("\n📋 系統資訊:")
    print(f"作業系統: {os.name}")
    print(f"工作目錄: {os.getcwd()}")
    print(f"Python 版本: {sys.version.split()[0]}")
    
    try:
        print(f"\n📋 套件版本:")
        print(f"OpenCV 版本: {cv2.__version__}")
        print(f"MediaPipe 版本: {mp.__version__}")
        print(f"NumPy 版本: {np.__version__}")
    except Exception as e:
        print(f"⚠️ 無法獲取版本信息: {e}")

    # 檢查必要的目錄
    print("\n📂 目錄檢查:")
    if not os.path.exists('analysis_reports'):
        try:
            os.makedirs('analysis_reports')
            print("✅ 已創建 analysis_reports 目錄")
        except Exception as e:
            print(f"❌ 無法創建 analysis_reports 目錄: {e}")
    else:
        print("✅ analysis_reports 目錄已存在")

    arm_tracker = ArmAxisTracker()
    
    # 處理影片路徑輸入
    print("\n📹 請輸入影片路徑")
    print("提示：")
    print("1. 可以使用相對路徑（例如：videos/test.mp4）")
    print("2. 或使用絕對路徑（例如：C:/Users/YourName/Videos/test.mp4）")
    print("3. 支援的格式：MP4, AVI, MOV 等")
    print("4. 按 Ctrl+C 可以隨時退出程式")
    print("-" * 50)
    
    while True:
        try:
            video_path = input("\n請輸入影片路徑: ").strip()
            if not video_path:
                print("❌ 錯誤：路徑不能為空")
                continue
                
            # 移除引號（如果有的話）
            video_path = video_path.strip('"\'')
            
            if arm_tracker.load_video(video_path):
                break
            else:
                print("\n請重新輸入影片路徑，或按 Ctrl+C 退出程式")
        except KeyboardInterrupt:
            print("\n\n👋 程式已終止")
            return
        except Exception as e:
            print(f"\n❌ 發生錯誤: {e}")
            print("請重新輸入影片路徑")

    print("\n🎬 開始分析影片...")
    print("📝 控制說明:")
    print("p: 暫停/恢復")
    print("r: 重置")
    print("s: 開始/停止記錄")
    print("e: 匯出分析報告")
    print("[: 減慢播放速度")
    print("]: 加快播放速度")
    print("ESC: 退出")
    print("-" * 50)

    try:
        print("\n🤖 正在初始化 MediaPipe Pose...")
        print("✅ MediaPipe Pose 初始化成功")
        
        paused = False
        recording = False
        analysis_complete = False

        while True:
            if not paused:
                ret, frame = arm_tracker.cap.read()
                if not ret:
                    print("\n✅ 影片播放完成")
                    analysis_complete = True
                    break

                processed_frame = arm_tracker.process_video_frame(frame)
                if processed_frame is not None:
                    cv2.imshow('Arm Axis Tracker', processed_frame)
                    arm_tracker.current_frame += 1

            wait_time = max(1, int(1000/arm_tracker.video_fps/arm_tracker.playback_speed))
            key = cv2.waitKey(wait_time) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r'):  # Reset
                arm_tracker.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                arm_tracker.current_frame = 0
            elif key == ord('p'):  # Pause/Resume
                paused = not paused
                print("⏸ 暫停" if paused else "▶️ 恢復")
            elif key == ord('s'):  # Start/Stop Recording
                if not arm_tracker.recording:
                    arm_tracker.start_recording()
                else:
                    arm_tracker.stop_recording()
            elif key == ord('e'):  # Export Report
                arm_tracker.export_analysis_report()
            elif key == ord('['):  # Slow down
                arm_tracker.playback_speed = max(0.25, arm_tracker.playback_speed - 0.25)
                print(f"播放速度: {arm_tracker.playback_speed}x")
            elif key == ord(']'):  # Speed up
                arm_tracker.playback_speed = min(2.0, arm_tracker.playback_speed + 0.25)
                print(f"播放速度: {arm_tracker.playback_speed}x")

        # 影片播放完成後，詢問是否要匯出報告
        if analysis_complete:
            print("\n📊 分析完成！")
            export = input("是否要匯出分析報告？(y/n): ").lower()
            if export == 'y':
                arm_tracker.export_analysis_report()
                print("✅ 報告已匯出")

    except Exception as e:
        print(f"\n❌ 主程式崩潰: {e}")
        import traceback
        traceback.print_exc()
    finally:
        arm_tracker.release_resources()
        cv2.destroyAllWindows()
        print("\n🧹 資源已釋放")
        print("👋 程式結束")

if __name__ == "__main__":
    main()