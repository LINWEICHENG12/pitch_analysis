#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
髖關節移動軌跡追蹤系統
專門用於分析面向攝影機的人物髖關節水平移動模式
適用於投球動作分析
"""

import cv2
import mediapipe as mp
import time
import math
import os
import sys
from collections import deque

class HipTrajectoryTracker:
    """髖關節軌跡追蹤器（只計算水平移動，且在幾乎不動時暫停記錄）"""

    def __init__(self, max_history=300, stationary_px_threshold=5):
        self.max_history = max_history
        # 閾值：如果水平位移（像素）低於此值，視為未移動
        self.stationary_px_threshold = stationary_px_threshold

        # 髖關節軌跡記錄
        self.center_hip_trajectory = deque(maxlen=max_history)

        # 時間戳記錄
        self.timestamps = deque(maxlen=max_history)

        # 移動統計（只累計水平距離）
        self.total_displacement = 0.0   # 累計水平位移（公尺）
        self.max_speed = 0.0            # 水平最大速度（m/s）
        self.speeds = deque(maxlen=50)  # 水平速度（m/s）緩衝

        # 校正參數：像素 / 公尺
        self.pixels_per_meter = 300  # 初始值，可由 calibrate_scale() 調整

        # 檢查系統環境
        self._check_system_environment()

        print("🎯 髖關節軌跡追蹤器初始化完成")
        print(f"📐 預設比例：每公尺 {self.pixels_per_meter} 像素")

    def _check_system_environment(self):
        """檢查系統環境和路徑（僅列印資訊，不影響邏輯）"""
        try:
            current_dir = os.getcwd()
            print(f"📁 當前工作目錄: {current_dir}")
        except Exception as e:
            print(f"⚠️ 環境檢查警告: {e}")

    def calibrate_scale(self, known_distance_pixels, known_distance_meters):
        """校正像素到公尺的比例"""
        if known_distance_meters <= 0:
            print("❌ 校正失敗：公尺距離需大於 0")
            return
        self.pixels_per_meter = known_distance_pixels / known_distance_meters
        print(f"✅ 校正完成：每公尺 {self.pixels_per_meter:.1f} 像素")

    def extract_hip_landmarks(self, results, frame_width, frame_height):
        """提取左右髖關節中心的座標（像素）"""
        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        lh = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        rh = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

        left_hip_px = (int(lh.x * frame_width), int(lh.y * frame_height))
        right_hip_px = (int(rh.x * frame_width), int(rh.y * frame_height))
        center_hip_px = (
            (left_hip_px[0] + right_hip_px[0]) // 2,
            (left_hip_px[1] + right_hip_px[1]) // 2
        )
        return center_hip_px

    def add_hip_position(self, center_px, timestamp):
        """
        添加髖關節位置：
        - 僅在水平位移超過閾值時，才 append 且計算速度、累加距離。
        - 否則視為「靜止」，暫停記錄。
        """
        if not self.center_hip_trajectory:
            # 第一次進入，無須比較，直接加入
            self.center_hip_trajectory.append(center_px)
            self.timestamps.append(timestamp)
            return

        prev_x, prev_y = self.center_hip_trajectory[-1]
        curr_x, curr_y = center_px

        dx_px = abs(curr_x - prev_x)
        # 只用水平位移判斷是否移動
        if dx_px < self.stationary_px_threshold:
            # 水平位移在閾值之內，視為未移動，跳過 append
            return

        # 水平位移超過閾值，才記錄新點
        self.center_hip_trajectory.append(center_px)
        self.timestamps.append(timestamp)
        # 計算水平速度與累計水平距離
        self._calculate_horizontal_movement(prev_x, curr_x)

    def _calculate_horizontal_movement(self, prev_x, curr_x):
        """計算水平速度與累加水平距離，並更新最大速度"""
        dx_px = abs(curr_x - prev_x)
        # 像素轉公尺
        dx_m = dx_px / self.pixels_per_meter

        # 取最後兩個 timestamp 計算 dt
        t_now = self.timestamps[-1]
        t_prev = self.timestamps[-2]
        dt = t_now - t_prev
        if dt <= 0:
            return

        speed_ms = dx_m / dt
        self.speeds.append(speed_ms)
        if speed_ms > self.max_speed:
            self.max_speed = speed_ms

        self.total_displacement += dx_m

    def get_movement_statistics(self):
        """回傳水平移動統計資料（速度、距離等）"""
        if not self.speeds:
            return None
        avg_speed = sum(self.speeds) / len(self.speeds)
        return {
            'current_speed': self.speeds[-1],
            'average_speed': avg_speed,
            'max_speed': self.max_speed,
            'total_horizontal_distance': self.total_displacement,
            'trajectory_points': len(self.center_hip_trajectory)
        }

    def draw_trajectory(self, frame, max_points=300):
        """繪製髖關節中心的水平軌跡（保留既有軌跡）"""
        pts = list(self.center_hip_trajectory)[-max_points:]
        if len(pts) < 2:
            return frame

        for i in range(1, len(pts)):
            prev_pt = pts[i - 1]
            curr_pt = pts[i]
            # 顏色可依據索引漸變
            ratio = i / len(pts)
            color = (int(255 * ratio), int(255 * (1 - ratio)), 128)
            cv2.line(frame, prev_pt, curr_pt, color, 2)

        # 畫當前位置
        cx, cy = pts[-1]
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
        cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2)
        return frame

    def draw_statistics(self, frame):
        """在影像上顯示水平移動統計資訊"""
        stats = self.get_movement_statistics()
        if not stats:
            cv2.putText(frame, "No movement data", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.6
        th = 2
        lh = 25

        # 背景框
        cv2.rectangle(frame, (10, 10), (400, 160), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 160), (255, 255, 255), 2)

        y = 35
        cv2.putText(frame, "Horizontal Movement Stats", (15, y), font, 0.7, (0, 255, 255), 2)
        y += lh
        cv2.putText(frame, f"Curr Speed: {stats['current_speed']:.3f} m/s",
                    (15, y), font, fs, (255, 255, 255), th)
        y += lh
        cv2.putText(frame, f"Avg Speed: {stats['average_speed']:.3f} m/s",
                    (15, y), font, fs, (200, 200, 200), th)
        y += lh
        cv2.putText(frame, f"Max Speed: {stats['max_speed']:.3f} m/s",
                    (15, y), font, fs, (0, 255, 0), th)
        y += lh
        cv2.putText(frame, f"Total Hori Dist: {stats['total_horizontal_distance']:.3f} m",
                    (15, y), font, fs, (255, 255, 255), th)

        return frame

def open_camera_simple():
    """嘗試以索引 0 開啟攝影機，若失敗則回傳 None"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法以索引 0 開啟攝影機，請檢查裝置或驅動程式")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

def main():
    print("🚀 髖關節水平移動追蹤系統啟動")
    print("=" * 50)
    try:
        print(f"🐍 Python 版本: {sys.version.split()[0]}")
        print(f"📋 OpenCV 版本: {cv2.__version__}")
        print(f"🧠 MediaPipe 版本: {mp.__version__}")
    except:
        pass

    print("控制說明:")
    print("  'c' - 校正距離   'r' - 重置軌跡   's' - 顯示統計   'ESC' - 退出")
    print("=" * 50)

    # 初始化 Mediapipe Holistic
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    # 初始化追蹤器：stationary_px_threshold 可以自行調整
    hip_tracker = HipTrajectoryTracker(stationary_px_threshold=5)

    cap = open_camera_simple()
    if cap is None:
        return

    # 根據解析度調整校正參數
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if actual_width >= 1280:
        hip_tracker.pixels_per_meter = 400
        print("🎯 標準解析度模式：每公尺 400 像素")
    else:
        hip_tracker.pixels_per_meter = 300
        print("🎯 基本解析度模式：每公尺 300 像素")

    try:
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ 無法讀取影像，結束")
                    break

                frame = cv2.flip(frame, 1)  # 左右翻轉鏡像
                current_time = time.time()

                # Mediapipe 姿態偵測
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = holistic.process(rgb)
                rgb.flags.writeable = True

                display = frame.copy()
                if results.pose_landmarks:
                    # 畫骨架
                    mp_drawing.draw_landmarks(
                        display,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
                    )

                    center_px = hip_tracker.extract_hip_landmarks(
                        results, display.shape[1], display.shape[0]
                    )
                    if center_px:
                        # 只在水平移動超過閾值時才記錄
                        hip_tracker.add_hip_position(center_px, current_time)

                    # 畫當前髖關節中心點
                    if hip_tracker.center_hip_trajectory:
                        cx, cy = hip_tracker.center_hip_trajectory[-1]
                        cv2.circle(display, (cx, cy), 6, (0, 0, 255), -1)
                        cv2.putText(display, "Hip", (cx + 5, cy - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # 繪製軌跡與統計
                    display = hip_tracker.draw_trajectory(display)
                    display = hip_tracker.draw_statistics(display)

                # 顯示控制提示
                cv2.putText(display, "c:Calibrate   r:Reset   s:Stats   ESC:Exit",
                            (10, display.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Hip Horizontal Tracker", display)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('c'):
                    print("\n=== 距離校正 ===")
                    print("請在終端輸入：已知距離(公尺) 與 對應的像素長度")
                    vals = input("格式 → [公尺] [像素] ：").strip().split()
                    if len(vals) == 2:
                        try:
                            km = float(vals[0])
                            kp = float(vals[1])
                            hip_tracker.calibrate_scale(kp, km)
                        except:
                            print("❌ 輸入錯誤")
                    else:
                        print("❌ 輸入數量不符")
                elif key == ord('r'):
                    hip_tracker = HipTrajectoryTracker(stationary_px_threshold=5)
                    print("🔄 軌跡已重置")
                elif key == ord('s'):
                    stats = hip_tracker.get_movement_statistics()
                    print("\n" + "=" * 40)
                    print("📊 水平移動統計")
                    print("=" * 40)
                    if stats:
                        for k, v in stats.items():
                            print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
                    else:
                        print("無任何數據")
                    print("=" * 40)

    except Exception as e:
        print(f"❌ 程式崩潰: {e}")

    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("🧹 資源已釋放")

    final_stats = hip_tracker.get_movement_statistics()
    if final_stats:
        print("\n🎯 最終統計：")
        print(f"最大水平速度: {final_stats['max_speed']:.3f} m/s")
        print(f"總水平移動距離: {final_stats['total_horizontal_distance']:.3f} m")

if __name__ == "__main__":
    main()
