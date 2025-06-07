#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試系統是否可以正常運行
"""

import os
import sys

def test_imports():
    """測試所有必要的模組匯入"""
    try:
        print("🔍 測試模組匯入...")
        
        # 基本套件
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
        
        import mediapipe as mp
        print("✅ MediaPipe: 可用")
        
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
        
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        # 分析模組
        from arm.elbow_angle_analysis import ElbowAngleAnalyzer
        print("✅ ElbowAngleAnalyzer")
        
        from arm.arm_speed_video_analysis import ArmAxisTracker
        print("✅ ArmAxisTracker")
        
        from joint_angle.joint_angle_analysis import JointAngleAnalyzer
        print("✅ JointAngleAnalyzer")
        
        from leg.leg_angle_analysis import LegAngleAnalyzer
        print("✅ LegAngleAnalyzer")
        
        from hip.hip_video_analysis import HipVideoAnalyzer
        print("✅ HipVideoAnalyzer")
        
        return True
        
    except ImportError as e:
        print(f"❌ 匯入失敗: {e}")
        return False

def test_initialization():
    """測試分析器初始化"""
    try:
        print("\n🔧 測試分析器初始化...")
        
        from arm.elbow_angle_analysis import ElbowAngleAnalyzer
        from arm.arm_speed_video_analysis import ArmAxisTracker
        from joint_angle.joint_angle_analysis import JointAngleAnalyzer
        from leg.leg_angle_analysis import LegAngleAnalyzer
        from hip.hip_video_analysis import HipVideoAnalyzer
        
        elbow_analyzer = ElbowAngleAnalyzer()
        print("✅ ElbowAngleAnalyzer 初始化成功")
        
        arm_tracker = ArmAxisTracker()
        print("✅ ArmAxisTracker 初始化成功")
        
        joint_analyzer = JointAngleAnalyzer()
        print("✅ JointAngleAnalyzer 初始化成功")
        
        leg_analyzer = LegAngleAnalyzer()
        print("✅ LegAngleAnalyzer 初始化成功")
        
        hip_analyzer = HipVideoAnalyzer()
        print("✅ HipVideoAnalyzer 初始化成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 初始化失敗: {e}")
        return False

def test_camera():
    """測試攝影機是否可用"""
    try:
        print("\n📹 測試攝影機...")
        import cv2
        
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ 攝影機可用且能讀取影像")
                print(f"📐 影像尺寸: {frame.shape}")
            else:
                print("⚠️ 攝影機已開啟但無法讀取影像")
            cap.release()
        else:
            print("❌ 無法開啟攝影機")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ 攝影機測試失敗: {e}")
        return False

def main():
    print("=== All-in-One 運動分析系統測試 ===\n")
    
    # 測試匯入
    if not test_imports():
        print("\n❌ 模組匯入測試失敗")
        return
    
    # 測試初始化
    if not test_initialization():
        print("\n❌ 初始化測試失敗")
        return
    
    # 測試攝影機
    test_camera()
    
    print("\n🎉 系統測試完成！")
    print("\n📝 使用說明:")
    print("1. 執行 'python allin_fixed.py' 開始分析")
    print("2. 輸入影片檔案路徑（支援 .mp4, .avi 等格式）")
    print("3. 使用以下按鍵操作:")
    print("   p: 暫停/恢復")
    print("   m: 標記最大肩外旋")
    print("   r: 標記出手角度")
    print("   f: 標記左腳落地")
    print("   e: 匯出報告和圖表")
    print("   ESC: 結束")

if __name__ == "__main__":
    main()
