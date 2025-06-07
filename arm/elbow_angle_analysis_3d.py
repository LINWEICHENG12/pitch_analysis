import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import os

class ElbowAngleAnalyzer3D:
    def __init__(self, max_history=300):
        self.max_history = max_history
        self.elbow_angles = deque(maxlen=max_history)
        self.elbow_velocities = deque(maxlen=max_history)
        self.time_history = deque(maxlen=max_history)
        self.wrist_trajectory = deque(maxlen=max_history)
        self.video_path = None
        self.video_fps = 0
        self.video_frame_count = 0
        self.current_frame = 0
        self.playback_speed = 1.0
        self.video_width = 0
        self.video_height = 0
        self.cap = None
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

    def calculate_elbow_angle(self, landmarks):
        """三維計算手肘角度"""
        shoulder = np.array([
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
        ])
        elbow = np.array([
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].z
        ])
        wrist = np.array([
            landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].z
        ])
        upper_arm = shoulder - elbow
        forearm = wrist - elbow
        cos_angle = np.dot(upper_arm, forearm) / (np.linalg.norm(upper_arm) * np.linalg.norm(forearm))
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
                elbow_angle = self.calculate_elbow_angle(landmarks)
                if len(self.elbow_angles) > 0:
                    time_diff = current_time - self.time_history[-1]
                    elbow_velocity = self.calculate_angular_velocity(
                        elbow_angle,
                        self.elbow_angles[-1],
                        time_diff
                    )
                    self.elbow_velocities.append(elbow_velocity)
                self.elbow_angles.append(elbow_angle)
                self.time_history.append(current_time)
                wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
                wrist_pos = (int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0]))
                self.wrist_trajectory.append(wrist_pos)
                for i in range(1, len(self.wrist_trajectory)):
                    cv2.line(image, self.wrist_trajectory[i-1], self.wrist_trajectory[i], (0, 255, 255), 2)
                elbow_x = int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * image.shape[1])
                elbow_y = int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * image.shape[0])
                text = f"{elbow_angle:.1f}°"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(image, 
                            (elbow_x - 5, elbow_y - text_height - 5),
                            (elbow_x + text_width + 5, elbow_y + 5),
                            (0, 0, 0),
                            -1)
                cv2.putText(image, 
                           text,
                           (elbow_x, elbow_y),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7,
                           (255, 255, 255),
                           2)
                image = self.draw_statistics(image)
            return image
        except Exception as e:
            print(f"❌ 處理影片幀時發生錯誤: {str(e)}")
            return frame

    def draw_statistics(self, image):
        if len(self.elbow_angles) > 0 and len(self.elbow_velocities) > 0:
            current_angle = self.elbow_angles[-1]
            current_velocity = self.elbow_velocities[-1]
            cv2.putText(image, f"Elbow Angle: {current_angle:.1f}°",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Elbow Velocity: {current_velocity:.1f} deg/s",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return image

    def export_analysis_report(self):
        if not self.elbow_angles or not self.elbow_velocities:
            print("❌ No data to export")
            return
        output_dir = 'analysis_reports'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f'elbow_angle_analysis3d_{timestamp}'
        min_length = min(len(self.time_history), len(self.elbow_angles), len(self.elbow_velocities))
        time_data = list(self.time_history)[:min_length]
        angle_data = list(self.elbow_angles)[:min_length]
        velocity_data = list(self.elbow_velocities)[:min_length]
        max_angle = max(angle_data)
        min_angle = min(angle_data)
        max_velocity = max(velocity_data)
        avg_velocity = sum(velocity_data) / len(velocity_data)
        plt.figure(figsize=(12, 6))
        plt.plot(time_data, angle_data, 'g-', label='Elbow Angle (3D)')
        plt.title(f'Elbow Angle Analysis (3D)\n'
                 f'Range: {min_angle:.1f}° to {max_angle:.1f}°\n'
                 f'Video Info: {self.video_width}x{self.video_height} @ {self.video_fps}fps',
                 fontsize=12, pad=20)
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('Angle (degrees)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{base_filename}_angle.png'), dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(12, 6))
        plt.plot(time_data, velocity_data, 'r-', label='Elbow Angular Velocity (3D)')
        plt.title(f'Elbow Angular Velocity Analysis (3D)\n'
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
        report_path = os.path.join(output_dir, f'{base_filename}_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Elbow Angle Analysis Report (3D)\n")
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
            f.write("\nTime Point Analysis:\n")
            for t, a, v in zip(time_data, angle_data, velocity_data):
                f.write(f"Time {t:.2f}s: Angle {a:.1f}° ({v:.1f} deg/s)\n")
        print(f"✅ Analysis report saved to: {output_dir}")
        print(f"📊 Generated charts:")
        print(f"   - {base_filename}_angle.png")
        print(f"   - {base_filename}_velocity.png")
        print(f"📝 Generated report: {base_filename}_report.txt")

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
    analyzer = ElbowAngleAnalyzer3D()
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
                    cv2.imshow('Elbow Angle Analyzer 3D', processed_frame)
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