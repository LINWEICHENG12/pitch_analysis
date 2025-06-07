import cv2
import os
import numpy as np
from arm.elbow_angle_analysis import ElbowAngleAnalyzer
from arm.arm_speed_video_analysis import ArmAxisTracker
from joint_angle.joint_angle_analysis import JointAngleAnalyzer
from leg.leg_angle_analysis import LegAngleAnalyzer
from hip.hip_video_analysis import HipVideoAnalyzer
import matplotlib.pyplot as plt
from datetime import datetime

# 主流程

def main():
    print("\n=== All-in-One 運動分析系統 ===\n")
    print("請輸入影片路徑：")
    video_path = input().strip().strip('"\'')
    if not os.path.exists(video_path):
        print("❌ 找不到影片檔案")
        return

    # 初始化各分析器
    print("\n初始化分析模組...")
    elbow_analyzer = ElbowAngleAnalyzer()
    arm_tracker = ArmAxisTracker()
    joint_analyzer = JointAngleAnalyzer()
    leg_analyzer = LegAngleAnalyzer()
    hip_analyzer = HipVideoAnalyzer()

    # 載入影片
    print("\n載入影片...")
    elbow_analyzer.load_video(video_path)
    arm_tracker.load_video(video_path)
    joint_analyzer.load_video(video_path)
    leg_analyzer.load_video(video_path)
    hip_analyzer.load_video(video_path)

    print("\n影片載入完成，準備開始分析...")

    paused = False
    print("\n📝 控制說明:")
    print("p: 暫停/恢復  m: 標記最大肩外旋  r: 標記出手角度  f: 標記左腳落地  e: 匯出報告  ESC: 結束")
    print("-" * 50)

    last_frame = None
    video_ended = False
    # 新增：人工標記前腳落地時間
    left_foot_land_time = None
    while True:
        if not paused and not video_ended:
            # 讀取一幀
            ret, frame = elbow_analyzer.cap.read()
            if not ret:
                print("\n✅ 影片播放完成，仍可操作與標記，按 ESC 結束...")
                video_ended = True
                continue
            last_frame = frame.copy()
            # 同步給所有分析器
            elbow_analyzer.current_frame += 1
            arm_tracker.current_frame += 1
            joint_analyzer.current_frame += 1
            leg_analyzer.current_frame += 1
            hip_analyzer.current_frame += 1
            # 分析
            elbow_analyzer.process_frame(frame.copy())
            arm_img = arm_tracker.process_video_frame(frame.copy())
            joint_analyzer.process_frame(frame.copy())
            leg_analyzer.process_frame(frame.copy())
            hip_img = hip_analyzer.process_video_frame(frame.copy())
            # 疊加髖部軌跡到手掌軌跡畫面
            if arm_img is not None and hip_img is not None:
                display_img = hip_analyzer.draw_trajectory(arm_img.copy())
            elif arm_img is not None:
                display_img = arm_img
            elif hip_img is not None:
                display_img = hip_img
            else:
                display_img = frame
            cv2.imshow('All-in-One Analyzer', display_img)
        elif last_frame is not None:
            # 影片播完後，持續顯示最後一幀，允許操作
            arm_img = arm_tracker.process_video_frame(last_frame.copy())
            hip_img = hip_analyzer.process_video_frame(last_frame.copy())
            if arm_img is not None and hip_img is not None:
                display_img = hip_analyzer.draw_trajectory(arm_img.copy())
            elif arm_img is not None:
                display_img = arm_img
            elif hip_img is not None:
                display_img = hip_img
            else:
                display_img = last_frame
            cv2.imshow('All-in-One Analyzer', display_img)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            # 詢問是否匯出
            while True:
                choice = input("\n是否要建立新的圖表資料？(y/n): ").strip().lower()
                if choice in ['y', 'n']:
                    break
                print("❌ 請輸入 'y' 或 'n'")
            if choice == 'y':
                # 在匯出圖表和報告的部分加入錯誤處理和日誌記錄
                try:
                    print("\n📝 開始匯出圖表與報告...")
                    output_dir = 'analysis_reports'
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    base_filename = f'allin_report_{timestamp}'

                    # 確保數據長度一致
                    min_len = min(
                        len(joint_analyzer.time_history),
                        len(joint_analyzer.shoulder_rotation_velocities),
                        len(joint_analyzer.hip_abduction_velocities),
                        len(leg_analyzer.right_knee_velocities),
                        len(elbow_analyzer.elbow_velocities)
                    )
                    if min_len == 0:
                        raise ValueError("分析數據不足，無法生成圖表和報告。")

                    t = list(joint_analyzer.time_history)[:min_len]
                    shoulder_vel = list(joint_analyzer.shoulder_rotation_velocities)[:min_len]
                    hip_vel = list(joint_analyzer.hip_abduction_velocities)[:min_len]
                    knee_vel = list(leg_analyzer.right_knee_velocities)[:min_len]
                    elbow_vel = list(elbow_analyzer.elbow_velocities)[:min_len]

                    # 匯出圖表
                    plt.figure(figsize=(15, 10))
                    plt.subplot(2, 1, 1)
                    plt.plot(t, shoulder_vel, label='肩旋轉角速度')
                    plt.plot(t, hip_vel, label='髖旋轉角速度')
                    plt.plot(t, knee_vel, label='右膝角速度')
                    plt.plot(t, elbow_vel, label='手肘角速度')
                    plt.legend()
                    plt.title('角速度分析圖')
                    plt.xlabel('時間 (s)')
                    plt.ylabel('角速度 (deg/s)')
                    plt.grid(True)
                    plt.savefig(os.path.join(output_dir, f'{base_filename}_angular_velocity.png'))
                    plt.close()

                    # 匯出報告
                    report_path = os.path.join(output_dir, f'{base_filename}_report.txt')
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write("All-in-One 運動分析報告\n")
                        f.write("="*50 + "\n\n")
                        f.write(f"最大肩旋轉角速度: {max(shoulder_vel):.2f} deg/s\n")
                        f.write(f"最大髖旋轉角速度: {max(hip_vel):.2f} deg/s\n")
                        f.write(f"最大右膝角速度: {max(knee_vel):.2f} deg/s\n")
                        f.write(f"最大手肘角速度: {max(elbow_vel):.2f} deg/s\n")
                    print(f"✅ 圖表與報告已匯出至 {output_dir}")

                except Exception as e:
                    print(f"❌ 匯出失敗: {e}")
            break
        elif key == ord('p'):
            paused = not paused
            print("⏸ 暫停" if paused else "▶️ 恢復")
        elif key == ord('m') and paused:
            # 標記最大肩外旋
            if len(joint_analyzer.shoulder_rotation_angles) > 0:
                joint_analyzer.manual_max_external_rotation = joint_analyzer.shoulder_rotation_angles[-1]
                print(f"✅ 手動標記最大肩外旋角度: {joint_analyzer.manual_max_external_rotation:.1f}°")
            else:
                print("⚠️ 尚未偵測到肩外旋角度，無法標記")
        elif key == ord('r') and paused:
            # 標記出手角度
            if len(elbow_analyzer.elbow_angles) > 0 and len(elbow_analyzer.time_history) > 0:
                elbow_analyzer.release_angle = elbow_analyzer.elbow_angles[-1]
                elbow_analyzer.release_time = elbow_analyzer.time_history[-1]
                print(f"✅ 出手時手肘角度已記錄: {elbow_analyzer.release_angle:.1f}° (時間: {elbow_analyzer.release_time:.2f}s)")
            else:
                print("⚠️ 尚未偵測到手肘角度，無法標記")
        elif key == ord('f') and paused:
            # 標記前腳（左腳）落地
            if len(leg_analyzer.time_history) > 0:
                left_foot_land_time = leg_analyzer.time_history[-1]
                print(f"✅ 前腳（左腳）落地時間已記錄: {left_foot_land_time:.2f}s")
            else:
                print("⚠️ 尚未偵測到左腳時間，無法標記")
        elif key == ord('e'):
            # 匯出綜合圖表與報告
            print("📝 匯出綜合圖表與報告...")
            output_dir = 'analysis_reports'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f'allin_report_{timestamp}'

            # 1. 圖表1：肩旋轉角速度、髖旋轉角速度、右膝角速度、手肘角速度
            # 取最短長度對齊
            min_len = min(len(joint_analyzer.time_history),
                          len(joint_analyzer.shoulder_rotation_velocities),
                          len(joint_analyzer.hip_abduction_velocities),
                          len(leg_analyzer.right_knee_velocities),
                          len(elbow_analyzer.elbow_velocities))
            t = list(joint_analyzer.time_history)[:min_len]
            shoulder_vel = list(joint_analyzer.shoulder_rotation_velocities)[:min_len]
            hip_vel = list(joint_analyzer.hip_abduction_velocities)[:min_len]
            knee_vel = list(leg_analyzer.right_knee_velocities)[:min_len]
            elbow_vel = list(elbow_analyzer.elbow_velocities)[:min_len]
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 1, 1)
            if joint_analyzer.time_history and joint_analyzer.shoulder_rotation_velocities:
                plt.plot(joint_analyzer.time_history, joint_analyzer.shoulder_rotation_velocities, label='肩旋轉角速度')
            if hip_analyzer.time_history and hip_analyzer.hip_rotation_velocities:
                plt.plot(hip_analyzer.time_history, hip_analyzer.hip_rotation_velocities, label='髖旋轉角速度')
            if leg_analyzer.time_history and leg_analyzer.right_knee_velocities:
                plt.plot(leg_analyzer.time_history, leg_analyzer.right_knee_velocities, label='右膝角速度')
            if elbow_analyzer.time_history and elbow_analyzer.elbow_velocities:
                plt.plot(elbow_analyzer.time_history, elbow_analyzer.elbow_velocities, label='手肘角速度')
            # 標示最大肩外旋時間點
            if joint_analyzer.manual_max_external_rotation_time is not None:
                plt.axvline(x=joint_analyzer.manual_max_external_rotation_time, color='r', linestyle='--', label='最大肩外旋時間點')
            # 標示左腳落地時間點
            if left_foot_land_time is not None:
                plt.axvline(x=left_foot_land_time, color='g', linestyle='--', label='左腳落地時間點')
            # 標示球出手時間點
            if elbow_analyzer.release_time is not None:
                plt.axvline(x=elbow_analyzer.release_time, color='b', linestyle='--', label='球出手時間點')
            plt.legend()
            plt.title('肩旋轉角速度、髖旋轉角速度、右膝角速度、手肘角速度')
            plt.xlabel('時間 (s)')
            plt.ylabel('角速度 (deg/s)')
            plt.grid(True)
            plt.subplot(2, 1, 2)
            if arm_tracker.time_history and arm_tracker.wrist_accelerations:
                plt.plot(arm_tracker.time_history, arm_tracker.wrist_accelerations, label='手掌加速度')
            if hip_analyzer.time_history and hip_analyzer.hip_accelerations:
                plt.plot(hip_analyzer.time_history, hip_analyzer.hip_accelerations, label='髖部移動加速度')
            # 標示最大肩外旋時間點
            if joint_analyzer.manual_max_external_rotation_time is not None:
                plt.axvline(x=joint_analyzer.manual_max_external_rotation_time, color='r', linestyle='--', label='最大肩外旋時間點')
            # 標示左腳落地時間點
            if left_foot_land_time is not None:
                plt.axvline(x=left_foot_land_time, color='g', linestyle='--', label='左腳落地時間點')
            # 標示球出手時間點
            if elbow_analyzer.release_time is not None:
                plt.axvline(x=elbow_analyzer.release_time, color='b', linestyle='--', label='球出手時間點')
            plt.legend()
            plt.title('手掌加速度、髖部移動加速度')
            plt.xlabel('時間 (s)')
            plt.ylabel('加速度 (m/s²)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{base_filename}_charts.png'))
            plt.close()

            # 2. 圖表2：手掌加速度、髖部移動加速度
            # 手掌加速度
            wrist_speed = list(arm_tracker.wrist_speed_history)
            wrist_time = list(arm_tracker.time_history)
            wrist_acc = [0]
            for i in range(1, len(wrist_speed)):
                dt = wrist_time[i] - wrist_time[i-1]
                if dt > 1e-6:
                    acc = (wrist_speed[i] - wrist_speed[i-1]) / dt
                else:
                    acc = 0
                wrist_acc.append(acc)
            # 髖部加速度
            hip_speed = list(hip_analyzer.speed_history)
            hip_time = list(hip_analyzer.timestamps)
            hip_acc = [0]
            for i in range(1, len(hip_speed)):
                dt = hip_time[i] - hip_time[i-1]
                if dt > 1e-6:
                    acc = (hip_speed[i] - hip_speed[i-1]) / dt
                else:
                    acc = 0
                hip_acc.append(acc)
            # 取最短長度
            min_len2 = min(len(wrist_time), len(wrist_acc), len(hip_time), len(hip_acc))
            plt.figure(figsize=(12,6))
            plt.plot(wrist_time[:min_len2], wrist_acc[:min_len2], label='Wrist Acceleration (m/s²)')
            plt.plot(hip_time[:min_len2], hip_acc[:min_len2], label='Hip Acceleration (m/s²)')
            plt.title('Wrist vs Hip Acceleration')
            plt.xlabel('Time (s)')
            plt.ylabel('Acceleration (m/s²)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{base_filename}_acceleration.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 3. 報告：最大肩外旋角度、出手角度
            report_path = os.path.join(output_dir, f'{base_filename}_report.txt')
            # 更新報告生成邏輯，確保只包含用戶需要的文字紀錄
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"最大手掌速度: {max(arm_tracker.wrist_speed_history):.3f} m/s\n")
                f.write(f"髖部的水平移動位移: {hip_analyzer.total_displacement:.3f} m\n")
                f.write(f"最大肩內旋角速度: {max(joint_analyzer.shoulder_rotation_velocities):.1f} deg/s\n")
                f.write(f"左膝最大角速度: {max(leg_analyzer.left_knee_velocities):.1f} deg/s\n")
                f.write(f"肩膀與髖最大角度差異值: {max([abs(a - b) for a, b in zip(joint_analyzer.shoulder_girdle_rotations, joint_analyzer.hip_girdle_rotations)]):.1f} 度\n")
                if left_foot_land_time is not None:
                    f.write(f"人工標記前腳（左腳）落地時間: {left_foot_land_time:.2f}s\n")
                else:
                    f.write("人工標記前腳（左腳）落地時間: (未標記)\n")
                if joint_analyzer.manual_max_external_rotation is not None:
                    f.write(f"最大肩外旋角度: {joint_analyzer.manual_max_external_rotation:.1f}°\n")
                else:
                    f.write("最大肩外旋角度: (未標記)\n")
                if elbow_analyzer.release_angle is not None and elbow_analyzer.release_time is not None:
                    f.write(f"出手時手肘角度: {elbow_analyzer.release_angle:.1f}° (時間: {elbow_analyzer.release_time:.2f}s)\n")
                else:
                    f.write("出手時手肘角度: (未標記)\n")
            print(f"✅ 圖表與報告已匯出至 {output_dir}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()