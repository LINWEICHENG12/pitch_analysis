import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import os

class ElbowAngleAnalyzer:
    def __init__(self, max_history=300):
        self.max_history = max_history
        
        # 手肘角度和角速度歷史
        self.elbow_angles = deque(maxlen=max_history)
        self.elbow_velocities = deque(maxlen=max_history)
        
        # 時間歷史
        self.time_history = deque(maxlen=max_history)
        
        # 影片相關參數
        self.video_path = None
        self.video_fps = 0
        self.video_frame_count = 0
        self.current_frame = 0
        self.playback_speed = 1.0
        self.video_width = 0
        self.video_height = 0
        self.cap = None
        
        # 初始化 MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 新增：球離手時的手肘角度與時間
        self.release_angle = None
        self.release_time = None

    def calculate_elbow_angle(self, landmarks):
        """計算手肘角度"""
        # 取得三點座標
        shoulder = np.array([
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        ])
        elbow = np.array([
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
        ])
        wrist = np.array([
            landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        ])
        # 以肘為端點，分別計算上臂與前臂向量
        upper_arm = shoulder - elbow
        forearm = wrist - elbow
        # 計算夾角
        cos_angle = np.dot(upper_arm, forearm) / (np.linalg.norm(upper_arm) * np.linalg.norm(forearm))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return angle

    def calculate_angular_velocity(self, current_angle, prev_angle, time_diff):
        """計算角速度（度/秒）"""
        if time_diff > 1e-6:
            return (current_angle - prev_angle) / time_diff
        return 0

    def process_frame(self, frame):
        """處理影片幀"""
        if frame is None:
            return None

        try:
            # 轉換顏色空間
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                # 繪製骨架
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # 計算當前時間
                current_time = self.current_frame / self.video_fps

                # 計算手肘角度和角速度
                elbow_angle = self.calculate_elbow_angle(results.pose_landmarks.landmark)
                if len(self.elbow_angles) > 0:
                    time_diff = current_time - self.time_history[-1]
                    elbow_velocity = self.calculate_angular_velocity(
                        elbow_angle,
                        self.elbow_angles[-1],
                        time_diff
                    )
                    self.elbow_velocities.append(elbow_velocity)
                self.elbow_angles.append(elbow_angle)

                # 記錄時間
                self.time_history.append(current_time)

                # 在手肘位置顯示即時角度
                elbow_x = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * image.shape[1])
                elbow_y = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * image.shape[0])
                
                # 繪製角度文字背景
                text = f"{elbow_angle:.1f}°"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(image, 
                            (elbow_x - 5, elbow_y - text_height - 5),
                            (elbow_x + text_width + 5, elbow_y + 5),
                            (0, 0, 0),
                            -1)
                
                # 繪製角度文字
                cv2.putText(image, 
                           text,
                           (elbow_x, elbow_y),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7,
                           (255, 255, 255),
                           2)

                # 繪製統計資訊
                image = self.draw_statistics(image)

            return image

        except Exception as e:
            print(f"❌ 處理影片幀時發生錯誤: {str(e)}")
            return frame

    def draw_statistics(self, image):
        """繪製統計資訊"""
        if len(self.elbow_angles) > 0 and len(self.elbow_velocities) > 0:
            current_angle = self.elbow_angles[-1]
            current_velocity = self.elbow_velocities[-1]
            
            # 繪製手肘角度和角速度資訊
            cv2.putText(image, f"Elbow Angle: {current_angle:.1f}°",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Elbow Velocity: {current_velocity:.1f} deg/s",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return image

    def export_analysis_report(self):
        """匯出分析報告"""
        if not self.elbow_angles or not self.elbow_velocities:
            print("❌ No data to export")
            return

        # 創建輸出目錄
        output_dir = 'analysis_reports'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 生成檔案名稱
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f'elbow_angle_analysis_{timestamp}'

        # 確保數據長度一致
        min_length = min(len(self.time_history), 
                        len(self.elbow_angles),
                        len(self.elbow_velocities))
        
        time_data = list(self.time_history)[:min_length]
        angle_data = list(self.elbow_angles)[:min_length]
        velocity_data = list(self.elbow_velocities)[:min_length]

        # 計算統計數據
        max_angle = max(angle_data)
        min_angle = min(angle_data)
        max_velocity = max(velocity_data)
        avg_velocity = sum(velocity_data) / len(velocity_data)

        # 繪製角度-時間圖表
        plt.figure(figsize=(12, 6))
        plt.plot(time_data, angle_data, 'g-', label='Elbow Angle')
        plt.title(f'Elbow Angle Analysis\n'
                 f'Range: {min_angle:.1f}° to {max_angle:.1f}°\n'
                 f'Video Info: {self.video_width}x{self.video_height} @ {self.video_fps}fps',
                 fontsize=12, pad=20)
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('Angle (degrees)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        # 添加統計資訊
        plt.text(0.02, 0.98, 
                f'Angle Range: {min_angle:.1f}° to {max_angle:.1f}°\n'
                f'Max Velocity: {max_velocity:.1f} deg/s\n'
                f'Avg Velocity: {avg_velocity:.1f} deg/s',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{base_filename}_angle.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 繪製角速度-時間圖表
        plt.figure(figsize=(12, 6))
        plt.plot(time_data, velocity_data, 'r-', label='Elbow Angular Velocity')
        plt.title(f'Elbow Angular Velocity Analysis\n'
                 f'Max Velocity: {max_velocity:.1f} deg/s\n'
                 f'Video Info: {self.video_width}x{self.video_height} @ {self.video_fps}fps',
                 fontsize=12, pad=20)
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('Angular Velocity (deg/s)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{base_filename}_velocity.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 生成文字報告
        report_path = os.path.join(output_dir, f'{base_filename}_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Elbow Angle Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Video Information:\n")
            f.write(f"Resolution: {self.video_width}x{self.video_height}\n")
            f.write(f"Frame Rate: {self.video_fps} FPS\n")
            f.write(f"Total Frames: {self.video_frame_count}\n")
            f.write(f"File Size: {os.path.getsize(self.video_path) / (1024*1024):.2f} MB\n\n")
            
            f.write("Movement Data:\n")
            f.write(f"Max Elbow Angle: {max_angle:.1f}°\n")
            f.write(f"Min Elbow Angle: {min_angle:.1f}°\n")
            f.write(f"Max Angular Velocity: {max_velocity:.1f} deg/s\n")
            f.write(f"Avg Angular Velocity: {avg_velocity:.1f} deg/s\n\n")
            
            # 新增：球離手時的手肘角度與時間
            if self.release_angle is not None and self.release_time is not None:
                f.write(f"球離手時手肘角度: {self.release_angle:.1f}° (時間: {self.release_time:.2f}s)\n\n")
            
            f.write("\nTime Point Analysis:\n")
            for i, (t, a, v) in enumerate(zip(time_data, angle_data, velocity_data)):
                f.write(f"Time {t:.2f}s: Angle {a:.1f}° ({v:.1f} deg/s)\n")

        print(f"✅ Analysis report saved to: {output_dir}")
        print(f"📊 Generated charts:")
        print(f"   - {base_filename}_angle.png")
        print(f"   - {base_filename}_velocity.png")
        print(f"📝 Generated report: {base_filename}_report.txt")

    def load_video(self, video_path):
        """載入影片檔案"""
        try:
            # 處理路徑
            video_path = os.path.abspath(video_path)
            print(f"📂 Loading video: {video_path}")
            
            if not os.path.exists(video_path):
                print(f"❌ Error: Video file not found '{video_path}'")
                return False

            self.video_path = video_path
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                print("❌ Error: Cannot open video")
                return False
                
            # 獲取影片資訊
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"📹 Video Info:")
            print(f"Resolution: {self.video_width}x{self.video_height}")
            print(f"Frame Rate: {self.video_fps} FPS")
            print(f"Total Frames: {self.video_frame_count}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading video: {str(e)}")
            return False

def main():
    analyzer = ElbowAngleAnalyzer()
    
    while True:
        try:
            video_path = input("\nEnter video path: ").strip()
            if not video_path:
                print("❌ Error: Path cannot be empty")
                continue
                
            video_path = video_path.strip('"\'')
            
            if analyzer.load_video(video_path):
                break
            else:
                print("\nPlease enter video path again, or press Ctrl+C to exit")
        except KeyboardInterrupt:
            print("\n\n👋 Program terminated")
            return
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please enter video path again")

    print("\n🎬 Starting video analysis...")
    print("📝 Controls:")
    print("p: Pause/Resume")
    print("r: Reset")
    print("e: Export analysis report")
    print("[: Slow down")
    print("]: Speed up")
    print("ESC: Exit")
    print("-" * 50)

    try:
        paused = False
        analysis_complete = False

        while True:
            if not paused:
                ret, frame = analyzer.cap.read()
                if not ret:
                    print("\n✅ Video playback complete")
                    analysis_complete = True
                    
                    # 詢問是否建立新圖表
                    while True:
                        choice = input("\n是否要建立新的圖表？(y/n): ").strip().lower()
                        if choice in ['y', 'n']:
                            break
                        print("❌ 請輸入 'y' 或 'n'")
                    
                    if choice == 'y':
                        analyzer.export_analysis_report()
                    
                    break

                processed_frame = analyzer.process_frame(frame)
                if processed_frame is not None:
                    cv2.imshow('Elbow Angle Analyzer', processed_frame)
                    analyzer.current_frame += 1

            # 等待鍵盤輸入，最小為1ms，避免高fps時卡住
            wait_time = max(1, int(1000/analyzer.video_fps/analyzer.playback_speed))
            key = cv2.waitKey(wait_time) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r'):  # Reset
                analyzer.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                analyzer.current_frame = 0
            elif key == ord('p'):  # Pause/Resume
                paused = not paused
                print("⏸ Paused" if paused else "▶️ Resumed")
            elif key == ord('e'):  # Export Report
                analyzer.export_analysis_report()
            elif key == ord('['):  # Slow down
                analyzer.playback_speed = max(0.25, analyzer.playback_speed - 0.25)
                print(f"Playback speed: {analyzer.playback_speed}x")
            elif key == ord(']'):  # Speed up
                analyzer.playback_speed = min(2.0, analyzer.playback_speed + 0.25)
                print(f"Playback speed: {analyzer.playback_speed}x")
            # 新增：球離手時記錄手肘角度
            elif key == ord('m') and paused:
                if len(analyzer.elbow_angles) > 0 and len(analyzer.time_history) > 0:
                    analyzer.release_angle = analyzer.elbow_angles[-1]
                    analyzer.release_time = analyzer.time_history[-1]
                    print(f"✅ 球離手時手肘角度已記錄: {analyzer.release_angle:.1f}° (時間: {analyzer.release_time:.2f}s)")
                else:
                    print("⚠️ 尚未偵測到手肘角度，無法標記")

    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
    finally:
        if analyzer.cap is not None:
            analyzer.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 