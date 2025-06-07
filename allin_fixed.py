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
import re

# 修復版主流程

def extract_max_wrist_speed():
    """Extract the maximum wrist speed from analysis report files."""
    report_dir = 'analysis_reports'
    max_wrist_speed = 0

    if not os.path.exists(report_dir):
        print(f"⚠️ Report directory '{report_dir}' does not exist.")
        return max_wrist_speed

    for filename in os.listdir(report_dir):
        if filename.endswith('_report.txt'):
            filepath = os.path.join(report_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                for line in file:
                    match = re.search(r'Wrist Speed ([\d\.]+) m/s', line)
                    if match:
                        speed = float(match.group(1))
                        max_wrist_speed = max(max_wrist_speed, speed)

    return max_wrist_speed

def export_charts_and_report(elbow_analyzer, arm_tracker, joint_analyzer, leg_analyzer, hip_analyzer, left_foot_land_time):
    """匯出圖表和報告的統一函數"""
    try:
        print("\n📝 開始匯出圖表與報告...")
        output_dir = 'analysis_reports'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f'allin_report_{timestamp}'

        # 設置 matplotlib 後端以避免顯示問題
        plt.style.use('default')

        # 修正數據長度不一致的問題
        try:
            # 確保所有數據長度一致
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

            # 僅匯出角速度綜合比較圖表
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 1, 1)
            plt.plot(t, shoulder_vel, label='Shoulder Rotation Velocity')
            plt.plot(t, hip_vel, label='Hip Rotation Velocity')
            plt.plot(t, knee_vel, label='Right Knee Velocity')
            plt.plot(t, elbow_vel, label='Elbow Velocity')

            if joint_analyzer.manual_max_external_rotation_time is not None:
                max_ext_rot_time = joint_analyzer.manual_max_external_rotation_time
                max_ext_rot_value = joint_analyzer.manual_max_external_rotation
                plt.axvline(x=max_ext_rot_time, color='red', linestyle='--', label='Max Shoulder External Rotation')
                plt.text(max_ext_rot_time, max_ext_rot_value, f'{max_ext_rot_value:.1f}°', color='red')

            if left_foot_land_time is not None:
                plt.axvline(x=left_foot_land_time, color='green', linestyle='--', label='Left Foot Landing Time')
                plt.text(left_foot_land_time, max(shoulder_vel), f'{left_foot_land_time:.2f}s', color='green')

            if elbow_analyzer.release_time is not None:
                release_time = elbow_analyzer.release_time
                release_angle = elbow_analyzer.release_angle
                plt.axvline(x=release_time, color='blue', linestyle='--', label='Ball Release')
                plt.text(release_time, release_angle, f'{release_angle:.1f}°', color='blue')

            plt.legend()
            plt.title('Angular Velocity Analysis Chart')
            plt.xlabel('Time (s)')
            plt.ylabel('Angular Velocity (deg/s)')
            plt.grid(True)
            angular_velocity_path = os.path.join(output_dir, f'{base_filename}_angular_velocity.png')
            plt.savefig(angular_velocity_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Angular velocity chart saved: {angular_velocity_path}")

            # 僅匯出文字檔紀錄
            report_path = os.path.join(output_dir, f'{base_filename}_report.txt')
            max_wrist_speed = max(arm_tracker.wrist_speed_history) if arm_tracker.wrist_speed_history else 0
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("All-in-One 運動分析報告\n")
                f.write("="*50 + "\n\n")
                f.write(f"最大肩內旋角速度: {max(joint_analyzer.shoulder_rotation_velocities):.1f} deg/s\n")
                f.write(f"左膝最大角速度: {max(leg_analyzer.left_knee_velocities):.1f} deg/s\n")
                f.write(f"肩膀與髖最大角度差異值: {max([abs(a - b) for a, b in zip(joint_analyzer.shoulder_girdle_rotations, joint_analyzer.hip_girdle_rotations)]):.1f} 度\n")
                f.write(f"最大手腕速度: {max_wrist_speed:.2f} m/s\n")
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
            print(f"✅ 分析報告已保存: {report_path}")

        except ValueError as ve:
            print(f"❌ 數據長度不一致: {ve}")
            return False

        except Exception as e:
            print(f"❌ 匯出圖表時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 確保資源釋放
        finally:
            cv2.destroyAllWindows()
            print("所有視窗已關閉，資源已釋放。")

    except Exception as e:
        print(f"❌ 匯出圖表時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n=== All-in-One 運動分析系統 (修復版) ===\n")
    print("請輸入影片路徑：")
    video_path = input().strip().strip('"\'')
    if not os.path.exists(video_path):
        print("❌ 找不到影片檔案")
        return

    # Initialize variables to ensure they are accessible in the finally block
    elbow_analyzer = None
    arm_tracker = None
    joint_analyzer = None
    leg_analyzer = None
    hip_analyzer = None
    left_foot_land_time = None

    try:
        # 初始化各分析器
        print("\n初始化分析模組...")
        elbow_analyzer = ElbowAngleAnalyzer()
        arm_tracker = ArmAxisTracker()
        joint_analyzer = JointAngleAnalyzer()
        leg_analyzer = LegAngleAnalyzer()
        hip_analyzer = HipVideoAnalyzer()
        print("✅ 所有分析模組初始化成功")

        # 載入影片
        print("\n載入影片...")
        elbow_analyzer.load_video(video_path)
        arm_tracker.load_video(video_path)
        joint_analyzer.load_video(video_path)
        leg_analyzer.load_video(video_path)
        hip_analyzer.load_video(video_path)
        print("✅ 影片載入完成")
        
        print("\n📝 控制說明:")
        print("p: 暫停/恢復  m: 標記最大肩外旋  r: 標記出手角度  f: 標記左腳落地  e: 匯出報告  ESC: 結束")
        print("-" * 50)

        paused = False
        last_frame = None
        video_ended = False
        playback_speed = 1  # Default playback speed multiplier
        
        # Main loop and processing logic
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
                
                # 顯示
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
                # 影片播完後，持續顯示最後一幀
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
            
            key = cv2.waitKey(int(30 / playback_speed)) & 0xFF
            
            if key == 27:  # ESC
                # 詢問是否匯出
                while True:
                    choice = input("\n是否要建立新的圖表資料？(y/n): ").strip().lower()
                    if choice in ['y', 'n']:
                        break
                    print("❌ 請輸入 'y' 或 'n'")
                
                if choice == 'y':
                    export_charts_and_report(elbow_analyzer, arm_tracker, joint_analyzer, 
                                           leg_analyzer, hip_analyzer, left_foot_land_time)
                break
                
            elif key == ord('p'):
                paused = not paused
                print("⏸ 暫停" if paused else "▶️ 恢復")
                
            elif key == ord('+') or key == ord('='):
                playback_speed = min(playback_speed + 0.5, 5.0)  # Increase speed, max 5x
                print(f"⏩ Playback speed: {playback_speed}x")
                
            elif key == ord('-') or key == ord('_'):
                playback_speed = max(playback_speed - 0.5, 0.5)  # Decrease speed, min 0.5x
                print(f"⏪ Playback speed: {playback_speed}x")
                
            elif key == ord('m') and paused:
                # 標記最大肩外旋
                if hasattr(joint_analyzer, 'shoulder_rotation_angles') and len(joint_analyzer.shoulder_rotation_angles) > 0:
                    joint_analyzer.manual_max_external_rotation = joint_analyzer.shoulder_rotation_angles[-1]
                    if hasattr(joint_analyzer, 'time_history') and len(joint_analyzer.time_history) > 0:
                        joint_analyzer.manual_max_external_rotation_time = joint_analyzer.time_history[-1]
                    print(f"✅ 手動標記最大肩外旋角度: {joint_analyzer.manual_max_external_rotation:.1f}°")
                else:
                    print("⚠️ 尚未偵測到肩外旋角度，無法標記")
                    
            elif key == ord('r') and paused:
                # 標記出手角度
                if hasattr(elbow_analyzer, 'elbow_angles') and len(elbow_analyzer.elbow_angles) > 0:
                    elbow_analyzer.release_angle = elbow_analyzer.elbow_angles[-1]
                    if hasattr(elbow_analyzer, 'time_history') and len(elbow_analyzer.time_history) > 0:
                        elbow_analyzer.release_time = elbow_analyzer.time_history[-1]
                    print(f"✅ 出手時手肘角度已記錄: {elbow_analyzer.release_angle:.1f}° (時間: {elbow_analyzer.release_time:.2f}s)")
                else:
                    print("⚠️ 尚未偵測到手肘角度，無法標記")
                    
            elif key == ord('f') and paused:
                # 標記前腳（左腳）落地
                if hasattr(leg_analyzer, 'time_history') and len(leg_analyzer.time_history) > 0:
                    left_foot_land_time = leg_analyzer.time_history[-1]
                    print(f"✅ 前腳（左腳）落地時間已記錄: {left_foot_land_time:.2f}s")
                else:
                    print("⚠️ 尚未偵測到左腳時間，無法標記")
                    
            elif key == ord('e'):
                # 匯出綜合圖表與報告
                export_charts_and_report(elbow_analyzer, arm_tracker, joint_analyzer, 
                                       leg_analyzer, hip_analyzer, left_foot_land_time)
                
    except Exception as e:
        print(f"❌ 主迴圈發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 確保釋放所有資源
        if elbow_analyzer and elbow_analyzer.cap:
            elbow_analyzer.cap.release()
        if arm_tracker:
            arm_tracker.release_resources()
        if joint_analyzer and joint_analyzer.cap:
            joint_analyzer.cap.release()
        if leg_analyzer and leg_analyzer.cap:
            leg_analyzer.cap.release()
        if hip_analyzer and hip_analyzer.cap:
            hip_analyzer.cap.release()
        print("✅ 所有分析模組資源已釋放")

if __name__ == "__main__":
    main()
