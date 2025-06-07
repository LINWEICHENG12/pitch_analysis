import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import os

class JointAngleAnalyzer:
    def __init__(self, max_history=300, pixels_per_meter=346.8):
        self.max_history = max_history
        self.pixels_per_meter = pixels_per_meter
        
        # 肩關節內旋角度和角速度歷史
        self.shoulder_rotation_angles = deque(maxlen=max_history)
        self.shoulder_rotation_velocities = deque(maxlen=max_history)
        
        # 髖關節外展角度和角速度歷史
        self.hip_abduction_angles = deque(maxlen=max_history)
        self.hip_abduction_velocities = deque(maxlen=max_history)
        
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

        # 在 __init__ 新增肩帶初始向量與歷史
        self.shoulder_girdle_initial_vec = None
        self.shoulder_girdle_rotations = deque(maxlen=max_history)

        # 在 __init__ 新增髖帶初始向量與歷史
        self.hip_girdle_initial_vec = None
        self.hip_girdle_rotations = deque(maxlen=max_history)

        # 新增：手動記錄最大肩外旋角度
        self.manual_max_external_rotation = None
        self.manual_max_external_rotation_time = None

    def calculate_shoulder_rotation(self, landmarks):
        """計算肱骨與肩膀（左右肩連線）的夾角，平行為0度，肱骨後伸為負"""
        r_shoulder = np.array([
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
        ])
        l_shoulder = np.array([
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
        ])
        r_elbow = np.array([
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].z
        ])
        # 基準向量：左右肩連線（左→右）
        shoulder_vec = r_shoulder - l_shoulder
        # 上臂向量：右肩→右肘
        upper_arm = r_elbow - r_shoulder
        # 只取 x, y 平面
        shoulder_vec_xy = shoulder_vec[:2]
        upper_arm_xy = upper_arm[:2]
        # 計算夾角
        angle = np.degrees(np.arctan2(
            upper_arm_xy[0]*shoulder_vec_xy[1] - upper_arm_xy[1]*shoulder_vec_xy[0],
            upper_arm_xy[0]*shoulder_vec_xy[0] + upper_arm_xy[1]*shoulder_vec_xy[1]
        ))
        # 角度正規化到 -180~180
        if angle > 180:
            angle -= 360
        elif angle < -180:
            angle += 360
        return angle

    def calculate_hip_rotation(self, landmarks):
        """計算髖關節旋轉角度（身體平面法向量與右大腿向量夾角，保留正負號）"""
        r_hip = np.array([
            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].z
        ])
        l_hip = np.array([
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y,
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].z
        ])
        r_shoulder = np.array([
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
        ])
        r_knee = np.array([
            landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].z
        ])
        # 計算骨盆平面法向量
        v1 = l_hip - r_hip
        v2 = r_shoulder - r_hip
        plane_normal = np.cross(v1, v2)
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        # 大腿向量
        thigh = r_knee - r_hip
        thigh = thigh / np.linalg.norm(thigh)
        # 夾角
        cos_angle = np.dot(thigh, plane_normal)
        angle = np.degrees(np.arcsin(np.clip(cos_angle, -1.0, 1.0)))
        return angle

    def calculate_angular_velocity(self, current_angle, prev_angle, time_diff):
        """計算角速度（度/秒）"""
        if time_diff > 0:
            return (current_angle - prev_angle) / time_diff
        return 0

    def calculate_shoulder_girdle_rotation(self, landmarks, initial_vector=None):
        """計算肩帶（左右肩連線）在x-y平面上的旋轉角度，相對於初始方向"""
        r_shoulder = np.array([
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        ])
        l_shoulder = np.array([
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        ])
        # 肩帶向量（左肩->右肩）
        shoulder_vec = r_shoulder - l_shoulder
        # 初始向量
        if initial_vector is None:
            return 0, shoulder_vec
        # 計算與初始向量的夾角
        shoulder_vec_norm = shoulder_vec / np.linalg.norm(shoulder_vec)
        initial_vec_norm = initial_vector / np.linalg.norm(initial_vector)
        dot = np.dot(shoulder_vec_norm, initial_vec_norm)
        dot = np.clip(dot, -1.0, 1.0)
        angle = np.degrees(np.arccos(dot))
        # 方向判斷（z軸外積）
        cross = np.cross(np.append(initial_vec_norm, 0), np.append(shoulder_vec_norm, 0))
        if cross[2] < 0:
            angle = -angle
        return angle, initial_vector

    def calculate_hip_girdle_rotation(self, landmarks, initial_vector=None):
        """計算髖帶（左右髖連線）在x-y平面上的旋轉角度，相對於初始方向"""
        r_hip = np.array([
            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y
        ])
        l_hip = np.array([
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
        ])
        # 髖帶向量（左髖->右髖）
        hip_vec = r_hip - l_hip
        # 初始向量
        if initial_vector is None:
            return 0, hip_vec
        # 計算與初始向量的夾角
        hip_vec_norm = hip_vec / np.linalg.norm(hip_vec)
        initial_vec_norm = initial_vector / np.linalg.norm(initial_vector)
        dot = np.dot(hip_vec_norm, initial_vec_norm)
        dot = np.clip(dot, -1.0, 1.0)
        angle = np.degrees(np.arccos(dot))
        # 方向判斷（z軸外積）
        cross = np.cross(np.append(initial_vec_norm, 0), np.append(hip_vec_norm, 0))
        if cross[2] < 0:
            angle = -angle
        return angle, initial_vector

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

                # 計算肩關節內旋角度和角速度
                shoulder_rotation = self.calculate_shoulder_rotation(results.pose_landmarks.landmark)
                if len(self.shoulder_rotation_angles) > 0:
                    time_diff = current_time - self.time_history[-1]
                    shoulder_velocity = self.calculate_angular_velocity(
                        shoulder_rotation,
                        self.shoulder_rotation_angles[-1],
                        time_diff
                    )
                    self.shoulder_rotation_velocities.append(shoulder_velocity)
                self.shoulder_rotation_angles.append(shoulder_rotation)

                # 計算髖關節外展角度和角速度
                hip_rotation = self.calculate_hip_rotation(results.pose_landmarks.landmark)
                if len(self.hip_abduction_angles) > 0:
                    time_diff = current_time - self.time_history[-1]
                    hip_velocity = self.calculate_angular_velocity(
                        hip_rotation,
                        self.hip_abduction_angles[-1],
                        time_diff
                    )
                    self.hip_abduction_velocities.append(hip_velocity)
                self.hip_abduction_angles.append(hip_rotation)

                # 計算肩帶水平旋轉角度
                if self.shoulder_girdle_initial_vec is None:
                    shoulder_girdle_angle, self.shoulder_girdle_initial_vec = self.calculate_shoulder_girdle_rotation(results.pose_landmarks.landmark, None)
                else:
                    shoulder_girdle_angle, _ = self.calculate_shoulder_girdle_rotation(results.pose_landmarks.landmark, self.shoulder_girdle_initial_vec)
                self.shoulder_girdle_rotations.append(shoulder_girdle_angle)
                # 顯示於左右肩中點
                r_shoulder_x = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * image.shape[1])
                r_shoulder_y = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * image.shape[0])
                l_shoulder_x = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image.shape[1])
                l_shoulder_y = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image.shape[0])
                mid_x = int((r_shoulder_x + l_shoulder_x) / 2)
                mid_y = int((r_shoulder_y + l_shoulder_y) / 2)
                text_girdle = f"S-girdle: {shoulder_girdle_angle:.1f}°"
                (text_width, text_height), _ = cv2.getTextSize(text_girdle, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(image, (mid_x - 5, mid_y - text_height - 5), (mid_x + text_width + 5, mid_y + 5), (0, 0, 0), -1)
                cv2.putText(image, text_girdle, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)

                # 計算髖帶水平旋轉角度
                if self.hip_girdle_initial_vec is None:
                    hip_girdle_angle, self.hip_girdle_initial_vec = self.calculate_hip_girdle_rotation(results.pose_landmarks.landmark, None)
                else:
                    hip_girdle_angle, _ = self.calculate_hip_girdle_rotation(results.pose_landmarks.landmark, self.hip_girdle_initial_vec)
                self.hip_girdle_rotations.append(hip_girdle_angle)
                # 顯示於左右髖中點
                r_hip_x = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * image.shape[1])
                r_hip_y = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * image.shape[0])
                l_hip_x = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * image.shape[1])
                l_hip_y = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * image.shape[0])
                mid_hip_x = int((r_hip_x + l_hip_x) / 2)
                mid_hip_y = int((r_hip_y + l_hip_y) / 2)
                text_hip_girdle = f"H-girdle: {hip_girdle_angle:.1f}°"
                (text_width, text_height), _ = cv2.getTextSize(text_hip_girdle, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(image, (mid_hip_x - 5, mid_hip_y - text_height - 5), (mid_hip_x + text_width + 5, mid_hip_y + 5), (0, 0, 0), -1)
                cv2.putText(image, text_hip_girdle, (mid_hip_x, mid_hip_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # 記錄時間
                self.time_history.append(current_time)

                # 繪製統計資訊
                image = self.draw_statistics(image)

            return image

        except Exception as e:
            print(f"❌ 處理影片幀時發生錯誤: {str(e)}")
            return frame

    def draw_statistics(self, image):
        """繪製統計資訊"""
        if len(self.shoulder_rotation_velocities) > 0 and len(self.hip_abduction_velocities) > 0:
            current_shoulder_velocity = self.shoulder_rotation_velocities[-1]
            current_hip_velocity = self.hip_abduction_velocities[-1]
            
            # 繪製肩關節資訊
            cv2.putText(image, f"Shoulder Rotation: {current_shoulder_velocity:.1f} deg/s",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 繪製髖關節資訊
            cv2.putText(image, f"Hip Abduction: {current_hip_velocity:.1f} deg/s",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return image

    # --- 新增：自動基準校正 ---
    def calibrate_angle_baseline(self, angle_list, baseline_frames=10):
        """取前baseline_frames幀平均作為基準，回傳校正後的角度序列"""
        if len(angle_list) < baseline_frames:
            baseline = np.mean(angle_list)
        else:
            baseline = np.mean(angle_list[:baseline_frames])
        return [a - baseline for a in angle_list], baseline

    def export_analysis_report(self):
        """匯出分析報告"""
        if not self.shoulder_rotation_velocities or not self.hip_abduction_velocities:
            print("❌ No data to export")
            return

        # 創建輸出目錄
        output_dir = 'analysis_reports'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 生成檔案名稱
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f'joint_angle_analysis_{timestamp}'

        # 確保數據長度一致
        min_length = min(len(self.time_history), 
                        len(self.shoulder_rotation_angles),
                        len(self.hip_abduction_angles),
                        len(self.shoulder_rotation_velocities),
                        len(self.hip_abduction_velocities))
        
        time_data = list(self.time_history)[:min_length]
        shoulder_angle_data = list(self.shoulder_rotation_angles)[:min_length]
        hip_angle_data = list(self.hip_abduction_angles)[:min_length]
        shoulder_velocity_data = list(self.shoulder_rotation_velocities)[:min_length]
        hip_velocity_data = list(self.hip_abduction_velocities)[:min_length]

        # --- 統一方向：右側旋轉為正 ---
        # 若有需要可在這裡反向（如：shoulder_angle_data = [-a for a in shoulder_angle_data]）
        # 目前已用 arcsin 並保留正負號，方向一致

        # --- 自動基準校正 ---
        shoulder_angle_data, shoulder_baseline = self.calibrate_angle_baseline(shoulder_angle_data)
        hip_angle_data, hip_baseline = self.calibrate_angle_baseline(hip_angle_data)

        # 計算統計數據
        max_shoulder_velocity = max(shoulder_velocity_data)
        max_hip_velocity = max(hip_velocity_data)
        avg_shoulder_velocity = sum(shoulder_velocity_data) / len(shoulder_velocity_data)
        avg_hip_velocity = sum(hip_velocity_data) / len(hip_velocity_data)
        
        max_shoulder_angle = max(shoulder_angle_data)
        min_shoulder_angle = min(shoulder_angle_data)
        max_hip_angle = max(hip_angle_data)
        min_hip_angle = min(hip_angle_data)

        # --- shoulder_girdle_rotations ---
        shoulder_girdle_data = list(self.shoulder_girdle_rotations)[:min_length]
        hip_girdle_data = list(self.hip_girdle_rotations)[:min_length]
        # 基準校正
        shoulder_girdle_data, girdle_baseline = self.calibrate_angle_baseline(shoulder_girdle_data)
        hip_girdle_data, hip_girdle_baseline = self.calibrate_angle_baseline(hip_girdle_data)

        # 角度也轉為 -180~+180 度
        def wrap_angle_deg(a):
            return (a + 180) % 360 - 180
        shoulder_girdle_data = [wrap_angle_deg(a) for a in shoulder_girdle_data]
        hip_girdle_data = [wrap_angle_deg(a) for a in hip_girdle_data]

        # 角速度計算改為模360度最短圓弧差
        def angle_diff_deg(a, b):
            d = a - b
            d = (d + 180) % 360 - 180
            return d

        shoulder_girdle_velocity = [0] + [
            angle_diff_deg(shoulder_girdle_data[i], shoulder_girdle_data[i-1]) / (time_data[i] - time_data[i-1])
            if (time_data[i] - time_data[i-1]) > 1e-6 else 0
            for i in range(1, len(shoulder_girdle_data))
        ]
        hip_girdle_velocity = [0] + [
            angle_diff_deg(hip_girdle_data[i], hip_girdle_data[i-1]) / (time_data[i] - time_data[i-1])
            if (time_data[i] - time_data[i-1]) > 1e-6 else 0
            for i in range(1, len(hip_girdle_data))
        ]

        # 計算統計數據
        max_shoulder_girdle_angle = max(shoulder_girdle_data)
        min_shoulder_girdle_angle = min(shoulder_girdle_data)
        max_hip_girdle_angle = max(hip_girdle_data)
        min_hip_girdle_angle = min(hip_girdle_data)
        max_shoulder_girdle_velocity = max(shoulder_girdle_velocity)
        min_shoulder_girdle_velocity = min(shoulder_girdle_velocity)
        max_hip_girdle_velocity = max(hip_girdle_velocity)
        min_hip_girdle_velocity = min(hip_girdle_velocity)

        # 繪製角度-時間圖表
        plt.figure(figsize=(12, 6))
        plt.plot(time_data, shoulder_girdle_data, 'orange', label='Shoulder Girdle')
        plt.plot(time_data, hip_girdle_data, 'purple', label='Hip Girdle')
        plt.title(f'Girdle Angle Analysis\n'
                 f'Shoulder Girdle: {min_shoulder_girdle_angle:.1f}° to {max_shoulder_girdle_angle:.1f}° | '
                 f'Hip Girdle: {min_hip_girdle_angle:.1f}° to {max_hip_girdle_angle:.1f}°\n'
                 f'Video Info: {self.video_width}x{self.video_height} @ {self.video_fps}fps',
                 fontsize=12, pad=20)
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('Angle (degrees)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{base_filename}_girdle_angle.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 繪製角速度-時間圖表
        plt.figure(figsize=(12, 6))
        plt.plot(time_data, shoulder_girdle_velocity, 'orange', label='Shoulder Girdle Velocity')
        plt.plot(time_data, hip_girdle_velocity, 'purple', label='Hip Girdle Velocity')
        plt.title(f'Girdle Angular Velocity Analysis\n'
                 f'Shoulder Girdle: {min_shoulder_girdle_velocity:.1f} to {max_shoulder_girdle_velocity:.1f} deg/s | '
                 f'Hip Girdle: {min_hip_girdle_velocity:.1f} to {max_hip_girdle_velocity:.1f} deg/s\n'
                 f'Video Info: {self.video_width}x{self.video_height} @ {self.video_fps}fps',
                 fontsize=12, pad=20)
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('Angular Velocity (deg/s)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{base_filename}_girdle_velocity.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 生成文字報告
        report_path = os.path.join(output_dir, f'{base_filename}_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Joint Angular Velocity Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Video Information:\n")
            f.write(f"Resolution: {self.video_width}x{self.video_height}\n")
            f.write(f"Frame Rate: {self.video_fps} FPS\n")
            f.write(f"Total Frames: {self.video_frame_count}\n")
            f.write(f"File Size: {os.path.getsize(self.video_path) / (1024*1024):.2f} MB\n\n")
            
            f.write("Movement Data:\n")
            f.write(f"Max Shoulder Rotation Velocity: {max_shoulder_velocity:.1f} deg/s\n")
            f.write(f"Avg Shoulder Rotation Velocity: {avg_shoulder_velocity:.1f} deg/s\n")
            f.write(f"Max Hip Abduction Velocity: {max_hip_velocity:.1f} deg/s\n")
            f.write(f"Avg Hip Abduction Velocity: {avg_hip_velocity:.1f} deg/s\n\n")
            
            f.write("Angle Range Analysis:\n")
            f.write(f"Shoulder Rotation Range: {min_shoulder_angle:.1f}° to {max_shoulder_angle:.1f}°\n")
            f.write(f"最大肩外旋角度（負值）: {min_shoulder_angle:.1f}°\n")
            f.write(f"Hip Abduction Range: {min_hip_angle:.1f}° to {max_hip_angle:.1f}°\n\n")
            
            # 新增：手動記錄最大肩外旋角度
            if self.manual_max_external_rotation is not None:
                f.write(f"手動標記最大肩外旋角度: {self.manual_max_external_rotation:.1f}°\n\n")
            
            f.write("\nTime Point Analysis:\n")
            for i, (t, sa, ha, sv, hv) in enumerate(zip(time_data, 
                                                     shoulder_angle_data,
                                                     hip_angle_data,
                                                     shoulder_velocity_data,
                                                     hip_velocity_data)):
                f.write(f"Time {t:.2f}s: Shoulder {sa:.1f}° ({sv:.1f} deg/s), Hip {ha:.1f}° ({hv:.1f} deg/s)\n")

        print(f"✅ Analysis report saved to: {output_dir}")
        print(f"📊 Generated charts:")
        print(f"   - {base_filename}_girdle_angle.png")
        print(f"   - {base_filename}_girdle_velocity.png")
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
    analyzer = JointAngleAnalyzer()
    
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
                    cv2.imshow('Joint Angle Analyzer', processed_frame)
                    analyzer.current_frame += 1

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
            # 新增：手動記錄最大肩外旋角度
            elif key == ord('m') and paused:
                if len(analyzer.shoulder_rotation_angles) > 0:
                    analyzer.manual_max_external_rotation = analyzer.shoulder_rotation_angles[-1]
                    analyzer.manual_max_external_rotation_time = analyzer.current_frame / analyzer.video_fps
                    print(f"✅ 手動標記最大肩外旋角度: {analyzer.manual_max_external_rotation:.1f}°")
                else:
                    print("⚠️ 尚未偵測到肩外旋角度，無法標記")

    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
    finally:
        if analyzer.cap is not None:
            analyzer.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()