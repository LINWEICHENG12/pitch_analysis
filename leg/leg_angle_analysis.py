import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import os

class LegAngleAnalyzer:
    def __init__(self, max_history=300):
        self.max_history = max_history
        # 右腳、左腳膝關節角度與角速度歷史
        self.right_knee_angles = deque(maxlen=max_history)
        self.right_knee_velocities = deque(maxlen=max_history)
        self.left_knee_angles = deque(maxlen=max_history)
        self.left_knee_velocities = deque(maxlen=max_history)
        self.time_history = deque(maxlen=max_history)
        # 影片參數
        self.video_path = None
        self.video_fps = 0
        self.video_frame_count = 0
        self.current_frame = 0
        self.playback_speed = 1.0
        self.video_width = 0
        self.video_height = 0
        self.cap = None
        # MediaPipe 初始化
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

    def calculate_knee_angle(self, hip, knee, ankle):
        """計算膝關節角度（大腿與小腿夾角，打直時180度）"""
        thigh = np.array([hip.x - knee.x, hip.y - knee.y, hip.z - knee.z])
        shank = np.array([ankle.x - knee.x, ankle.y - knee.y, ankle.z - knee.z])
        cos_angle = np.dot(thigh, shank) / (np.linalg.norm(thigh) * np.linalg.norm(shank))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return angle

    def calculate_angular_velocity(self, current_angle, prev_angle, time_diff):
        if time_diff > 1e-6:
            return (current_angle - prev_angle) / time_diff
        return 0

    def process_frame(self, frame):
        if frame is None:
            return None
        try:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                landmarks = results.pose_landmarks.landmark
                current_time = self.current_frame / self.video_fps
                # 右腳
                right_angle = self.calculate_knee_angle(
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                )
                if len(self.right_knee_angles) > 0:
                    time_diff = current_time - self.time_history[-1]
                    velocity = self.calculate_angular_velocity(
                        right_angle, self.right_knee_angles[-1], time_diff)
                    self.right_knee_velocities.append(velocity)
                self.right_knee_angles.append(right_angle)
                # 左腳
                left_angle = self.calculate_knee_angle(
                    landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                    landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                    landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
                )
                if len(self.left_knee_angles) > 0:
                    time_diff = current_time - self.time_history[-1]
                    velocity = self.calculate_angular_velocity(
                        left_angle, self.left_knee_angles[-1], time_diff)
                    self.left_knee_velocities.append(velocity)
                self.left_knee_angles.append(left_angle)
                # 時間
                self.time_history.append(current_time)
                # 顯示即時角度
                rknee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
                lknee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
                rx = int(rknee.x * image.shape[1])
                ry = int(rknee.y * image.shape[0])
                lx = int(lknee.x * image.shape[1])
                ly = int(lknee.y * image.shape[0])
                text_r = f"R-knee: {right_angle:.1f}°"
                text_l = f"L-knee: {left_angle:.1f}°"
                (tw, th), _ = cv2.getTextSize(text_r, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(image, (rx-5, ry-th-5), (rx+tw+5, ry+5), (0,0,0), -1)
                cv2.putText(image, text_r, (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                (tw, th), _ = cv2.getTextSize(text_l, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(image, (lx-5, ly-th-5), (lx+tw+5, ly+5), (0,0,0), -1)
                cv2.putText(image, text_l, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            return image
        except Exception as e:
            print(f"❌ 處理影片幀時發生錯誤: {str(e)}")
            return frame

    def export_analysis_report(self):
        if not self.right_knee_angles or not self.left_knee_angles:
            print("❌ No data to export")
            return
        output_dir = 'analysis_reports'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f'leg_angle_analysis_{timestamp}'
        min_length = min(len(self.time_history), len(self.right_knee_angles), len(self.left_knee_angles), len(self.right_knee_velocities), len(self.left_knee_velocities))
        time_data = list(self.time_history)[:min_length]
        right_angle_data = list(self.right_knee_angles)[:min_length]
        left_angle_data = list(self.left_knee_angles)[:min_length]
        right_velocity_data = list(self.right_knee_velocities)[:min_length]
        left_velocity_data = list(self.left_knee_velocities)[:min_length]
        # 角度圖
        plt.figure(figsize=(12,6))
        plt.plot(time_data, right_angle_data, 'b-', label='Right Knee')
        plt.plot(time_data, left_angle_data, 'g-', label='Left Knee')
        plt.title('Knee Angle Analysis')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{base_filename}_angle.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # 角速度圖
        plt.figure(figsize=(12,6))
        plt.plot(time_data, right_velocity_data, 'b-', label='Right Knee Velocity')
        plt.plot(time_data, left_velocity_data, 'g-', label='Left Knee Velocity')
        plt.title('Knee Angular Velocity Analysis')
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Velocity (deg/s)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{base_filename}_velocity.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # 產生文字數據紀錄檔
        report_path = os.path.join(output_dir, f'{base_filename}_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Leg Angle Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Video Information:\n")
            f.write(f"Resolution: {self.video_width}x{self.video_height}\n")
            f.write(f"Frame Rate: {self.video_fps} FPS\n")
            f.write(f"Total Frames: {self.video_frame_count}\n")
            f.write(f"File Size: {os.path.getsize(self.video_path) / (1024*1024):.2f} MB\n\n")
            f.write("Knee Angle Data (per frame):\n")
            f.write(f"{'Time(s)':>8} | {'R-Knee Angle':>12} | {'R-Knee Vel.':>12} | {'L-Knee Angle':>12} | {'L-Knee Vel.':>12}\n")
            f.write("-"*60 + "\n")
            for t, ra, rv, la, lv in zip(time_data, right_angle_data, right_velocity_data, left_angle_data, left_velocity_data):
                f.write(f"{t:8.3f} | {ra:12.2f} | {rv:12.2f} | {la:12.2f} | {lv:12.2f}\n")
        print(f"📝 Generated report: {base_filename}_report.txt")
        print(f"✅ Analysis report saved to: {output_dir}")
        print(f"📊 Generated charts:")
        print(f"   - {base_filename}_angle.png")
        print(f"   - {base_filename}_velocity.png")

    def load_video(self, video_path):
        try:
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
    analyzer = LegAngleAnalyzer()
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
                    cv2.imshow('Leg Angle Analyzer', processed_frame)
                    analyzer.current_frame += 1
            wait_time = max(1, int(1000/analyzer.video_fps/analyzer.playback_speed))
            key = cv2.waitKey(wait_time) & 0xFF
            if key == 27:
                break
            elif key == ord('r'):
                analyzer.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                analyzer.current_frame = 0
            elif key == ord('p'):
                paused = not paused
                print("⏸ Paused" if paused else "▶️ Resumed")
            elif key == ord('e'):
                analyzer.export_analysis_report()
            elif key == ord('['):
                analyzer.playback_speed = max(0.25, analyzer.playback_speed - 0.25)
                print(f"Playback speed: {analyzer.playback_speed}x")
            elif key == ord(']'):
                analyzer.playback_speed = min(2.0, analyzer.playback_speed + 0.25)
                print(f"Playback speed: {analyzer.playback_speed}x")
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
    finally:
        if analyzer.cap is not None:
            analyzer.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 