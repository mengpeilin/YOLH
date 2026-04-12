#!/usr/bin/env python3
import pyrealsense2 as rs
import yaml
import argparse


def read_intrinsics(serial_number: str = None):
    """读取 RealSense 相机内参"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 如果指定序列号，使用该相机
    if serial_number:
        config.enable_device(serial_number)
    
    # 启用 RGB 流（内参基于 RGB sensor）
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    print("正在连接相机...")
    profile = pipeline.start(config)
    
    # 获取 RGB sensor 的内参
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()
    
    device = profile.get_device()
    serial = device.get_info(rs.camera_info.serial_number)
    
    print(f"\n✅ 成功读取相机内参")
    print(f"相机序列号: {serial}")
    print(f"分辨率: {intr.width} x {intr.height}")
    
    # 输出内参
    intrinsic_list = [intr.fx, intr.fy, intr.ppx, intr.ppy]
    
    print(f"\n📋 内参 (YAML 格式):")
    print(f"camera_intrinsic: {intrinsic_list}")
    print(f"camera_serial: '{serial}'")
    
    print(f"\n📋 内参详细信息:")
    print(f"  fx (x 焦距):    {intr.fx:.4f}")
    print(f"  fy (y 焦距):    {intr.fy:.4f}")
    print(f"  cx (x 主点):    {intr.ppx:.4f}")
    print(f"  cy (y 主点):    {intr.ppy:.4f}")
    
    pipeline.stop()
    
    return {
        'camera_intrinsic': intrinsic_list,
        'camera_serial': serial
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='读取 RealSense 相机内参')
    parser.add_argument('--serial', type=str, default=None, 
                       help='相机序列号（可选，不指定则使用第一个可用相机）')
    args = parser.parse_args()
    
    try:
        read_intrinsics(args.serial)
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("请检查 RealSense 相机是否已连接")
