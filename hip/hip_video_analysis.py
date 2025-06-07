#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
髖關節移動軌跡追蹤系統 - 影片分析版
專門用於分析影片中人物髖關節水平移動模式
適用於投球動作分析
"""

import cv2
import mediapipe as mp
import time
import math
import os
import sys
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class HipVideoAnalyzer:
    """髖關節軌跡追蹤器（只計算水平移動，且在幾乎不動時暫停記錄）"""

    def __init__(self, max_history=300, stationary_px_threshold=5):
        self.max_history = max_history
        # 閾值：如果水平位移（像素）低於此值，視為未移動
        self.stationary_px_threshold = stationary_px_threshold

        # 髖關節軌跡記錄
        self.center_hip_trajectory = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
        self.speeds = deque(maxlen=max_history)
        self.speed_history = deque(maxlen=max_history)

        # 移動統計（只累計水平距離）
        self.total_displacement = 0.0   # 累計水平位移（公尺）
        self.max_speed = 0.0            # 水平最大速度（m/s）

        # 校正參數：像素 / 公尺
        self.pixels_per_meter = 296  # 初始值，可由 calibrate_scale() 調整

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
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        print("🎯 髖關節軌跡追蹤器初始化完成")
        print(f"📐 預設比例：每公尺 {self.pixels_per_meter} 像素")

    def load_video(self, video_path):
        """載入影片檔案"""
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

    def calibrate_scale(self, known_distance_pixels, known_distance_meters):
        """校正像素到公尺的比例"""
        if known_distance_meters <= 0:
            print("❌ 校正失敗：公尺距離需大於 0")
            return
        self.pixels_per_meter = known_distance_pixels / known_distance_meters
        print(f"✅ 校正完成：每公尺 {self.pixels_per_meter:.1f} 像素")

    def extract_hip_landmarks(self, results):
        """提取左右髖關節中心的座標（像素）"""
        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        lh = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        rh = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

        left_hip_px = (int(lh.x * self.video_width), int(lh.y * self.video_height))
        right_hip_px = (int(rh.x * self.video_width), int(rh.y * self.video_height))
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
        # 計算水平速度與累加水平距離
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
        self.speed_history.append(speed_ms)
        if speed_ms > self.max_speed:
            self.max_speed = speed_ms

        self.total_displacement += dx_m

    def get_movement_statistics(self):
        """回傳水平移動統計資料（速度、距離等）"""
        if not self.speeds:
            return None
        avg_speed = sum(self.speeds) / len(self.speeds)
        return {
            'current_speed': self.speeds[-1] if self.speeds else 0,
            'average_speed': avg_speed,
            'max_speed': self.max_speed,
            'total_horizontal_distance': self.total_displacement,
            'trajectory_points': len(self.center_hip_trajectory)
        }

    def draw_trajectory(self, frame):
        """繪製髖關節中心的水平軌跡（保留既有軌跡）"""
        pts = list(self.center_hip_trajectory)
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
        if pts:
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

    def process_video_frame(self, frame):
        """處理影片幀"""
        if frame is None:
            return None

        try:
            # 轉換顏色空間
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 繪製骨架
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_holistic.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
                )

                center_px = self.extract_hip_landmarks(results)
                if center_px:
                    # 只在水平移動超過閾值時才記錄
                    current_time = self.current_frame / self.video_fps
                    self.add_hip_position(center_px, current_time)

                # 繪製軌跡與統計
                image = self.draw_trajectory(image)
                image = self.draw_statistics(image)

            # 添加影片資訊
            cv2.putText(image, f"Frame: {self.current_frame}/{self.video_frame_count}",
                        (10, self.video_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"Time: {self.current_frame/self.video_fps:.2f}s",
                        (10, self.video_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"Speed: {self.playback_speed}x",
                        (10, self.video_height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            return image

        except Exception as e:
            print(f"❌ 處理影片幀時發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame

    def smooth_data(self, data, window_size=5):
        """使用移動平均平滑數據"""
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def export_analysis_report(self):
        """匯出分析報告"""
        try:
            if not self.speed_history or not self.timestamps:
                print("❌ 沒有可匯出的數據")
                return

            # 確保數據長度一致
            min_length = min(len(self.timestamps), len(self.speed_history))
            if min_length == 0:
                print("❌ 沒有可匯出的數據")
                return

            timestamps = list(self.timestamps)[:min_length]
            speed_history = list(self.speed_history)[:min_length]

            # 根據影片FPS調整平滑窗口大小
            window_size = max(3, min(7, int(self.video_fps / 10)))  # 降低平滑程度
            smoothed_speed = self.smooth_data(speed_history, window_size)
            
            # 調整時間戳以匹配平滑後的數據長度
            smoothed_timestamps = timestamps[window_size-1:]

            # 創建輸出目錄
            output_dir = 'analysis_reports'
            try:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
            except Exception as e:
                print(f"❌ 無法創建輸出目錄: {e}")
                return

            # 生成檔案名稱
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f'hip_analysis_report_{timestamp}'

            # 計算統計數據
            try:
                max_speed = max(speed_history)
                avg_speed = sum(speed_history)/len(speed_history)
            except Exception as e:
                print(f"❌ 計算統計數據時發生錯誤: {e}")
                return

            # 速度-時間圖表
            try:
                plt.figure(figsize=(12, 6))
                # 繪製原始數據（半透明）
                plt.plot(timestamps, speed_history, 'b-', alpha=0.3, label='原始速度')
                # 繪製平滑後的數據
                plt.plot(smoothed_timestamps, smoothed_speed, 'r-', label='平滑後速度')
                plt.title(f'髖關節水平移動速度分析\n'
                         f'最大速度: {max_speed:.2f} m/s | 平均速度: {avg_speed:.2f} m/s\n'
                         f'影片資訊: {self.video_width}x{self.video_height} @ {self.video_fps}fps',
                         fontsize=12, pad=20)
                plt.xlabel('時間 (秒)', fontsize=10)
                plt.ylabel('速度 (m/s)', fontsize=10)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(fontsize=10)
                
                # 添加統計資訊
                plt.text(0.02, 0.98, 
                        f'最大速度: {max_speed:.2f} m/s\n'
                        f'平均速度: {avg_speed:.2f} m/s\n'
                        f'總水平移動距離: {self.total_displacement:.2f} m\n'
                        f'平滑窗口大小: {window_size} 幀',
                        transform=plt.gca().transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{base_filename}_speed.png'), dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"❌ 生成圖表時發生錯誤: {e}")
                return

            # 生成文字報告
            try:
                report_path = os.path.join(output_dir, f'{base_filename}_report.txt')
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write("髖關節水平移動分析報告\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"影片資訊:\n")
                    f.write(f"解析度: {self.video_width}x{self.video_height}\n")
                    f.write(f"幀率: {self.video_fps} FPS\n")
                    f.write(f"總幀數: {self.video_frame_count}\n")
                    if self.video_path and os.path.exists(self.video_path):
                        f.write(f"檔案大小: {os.path.getsize(self.video_path) / (1024*1024):.2f} MB\n")
                    f.write("\n")
                    
                    f.write("運動數據:\n")
                    f.write(f"最大速度: {max_speed:.2f} m/s\n")
                    f.write(f"平均速度: {avg_speed:.2f} m/s\n")
                    f.write(f"總水平移動距離: {self.total_displacement:.2f} m\n")
                    f.write(f"平滑窗口大小: {window_size} 幀\n")
                    
                    f.write("\n時間點分析:\n")
                    for i, (t, s) in enumerate(zip(smoothed_timestamps, smoothed_speed)):
                        f.write(f"時間 {t:.2f}s: 速度 {s:.2f} m/s\n")
            except Exception as e:
                print(f"❌ 生成文字報告時發生錯誤: {e}")
                return

            print(f"✅ 分析報告已保存至: {output_dir}")
            print(f"📊 已生成圖表：")
            print(f"   - {base_filename}_speed.png")
            print(f"📝 已生成報告：{base_filename}_report.txt")

        except Exception as e:
            print(f"❌ 匯出分析報告時發生錯誤: {e}")
            import traceback
            traceback.print_exc()

def main():
    print("\n" + "=" * 50)
    print("🚀 髖關節水平移動分析系統")
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

    analyzer = HipVideoAnalyzer()
    
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
            
            if analyzer.load_video(video_path):
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
    print("e: 匯出分析報告")
    print("[: 減慢播放速度")
    print("]: 加快播放速度")
    print("c: 校正距離比例")
    print("ESC: 退出")
    print("-" * 50)

    try:
        print("\n🤖 正在初始化 MediaPipe Holistic...")
        print("✅ MediaPipe Holistic 初始化成功")
        
        paused = False
        analysis_complete = False

        while True:
            if not paused:
                ret, frame = analyzer.cap.read()
                if not ret:
                    print("\n✅ 影片播放完成")
                    analysis_complete = True
                    break

                processed_frame = analyzer.process_video_frame(frame)
                if processed_frame is not None:
                    cv2.imshow('Hip Horizontal Tracker', processed_frame)
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
                print("⏸ 暫停" if paused else "▶️ 恢復")
            elif key == ord('e'):  # Export Report
                analyzer.export_analysis_report()
            elif key == ord('['):  # Slow down
                analyzer.playback_speed = max(0.25, analyzer.playback_speed - 0.25)
                print(f"播放速度: {analyzer.playback_speed}x")
            elif key == ord(']'):  # Speed up
                analyzer.playback_speed = min(2.0, analyzer.playback_speed + 0.25)
                print(f"播放速度: {analyzer.playback_speed}x")
            elif key == ord('c'):  # Calibrate
                print("\n=== 距離校正 ===")
                print("請在終端輸入：已知距離(公尺) 與 對應的像素長度")
                vals = input("格式 → [公尺] [像素] ：").strip().split()
                if len(vals) == 2:
                    try:
                        km = float(vals[0])
                        kp = float(vals[1])
                        analyzer.calibrate_scale(kp, km)
                    except:
                        print("❌ 輸入錯誤")
                else:
                    print("❌ 輸入數量不符")

        # 影片播放完成後，詢問是否要匯出報告
        if analysis_complete:
            print("\n📊 分析完成！")
            export = input("是否要匯出分析報告？(y/n): ").lower()
            if export == 'y':
                analyzer.export_analysis_report()
                print("✅ 報告已匯出")

    except Exception as e:
        print(f"\n❌ 主程式崩潰: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if analyzer.cap is not None:
            analyzer.cap.release()
        cv2.destroyAllWindows()
        print("\n🧹 資源已釋放")
        print("👋 程式結束")

if __name__ == "__main__":
    main() 