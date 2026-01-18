#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import glob
import cv2
from tqdm import tqdm
from PIL import Image
import argparse
import numpy as np
from datetime import datetime

def yolo2coco(yolo_dataset_path, output_json_path, subset_file=None):
    """
    将YOLO格式的数据集转换为COCO格式
    
    Args:
        yolo_dataset_path: YOLO数据集的根目录
        output_json_path: 输出的COCO格式JSON文件路径
        subset_file: 可选，指定包含要处理的图像列表的文本文件路径
    """
    # 获取类别信息
    classes_file = os.path.join(yolo_dataset_path, "classes.txt")
    if not os.path.exists(classes_file):
        # 尝试在其他目录中查找classes.txt
        classes_file = os.path.join(yolo_dataset_path, "labels", "classes.txt")
        if not os.path.exists(classes_file):
            for root, dirs, files in os.walk(os.path.join(yolo_dataset_path, "labels")):
                if "classes.txt" in files:
                    classes_file = os.path.join(root, "classes.txt")
                    break
    
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"未找到classes.txt文件")
    
    # 读取类别信息
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    
    # 创建COCO格式的数据结构
    coco_data = {
        "info": {
            "description": "COCO_labels",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "categories": [],
        "images": [],
        "annotations": []
    }
    
    # 添加类别信息 - 从1开始
    for i, cls_name in enumerate(classes):
        coco_data["categories"].append({
            "id": i + 1,  # 修改点1：类别ID从1开始
            "name": cls_name,
            "supercategory": ""
        })
    
    # 创建类别名称到ID的映射
    class_name_to_id = {cls_name: i + 1 for i, cls_name in enumerate(classes)}
    
    # 获取所有图像文件
    image_files = []
    
    # 如果提供了子集文件，则从中读取图像列表
    if subset_file and os.path.exists(subset_file):
        print(f"使用子集文件: {subset_file}")
        with open(subset_file, 'r', encoding='utf-8') as f:
            file_paths = [line.strip() for line in f.readlines() if line.strip()]
            
        # 确保所有路径都存在
        image_files = [path for path in file_paths if os.path.exists(path)]
        if len(image_files) != len(file_paths):
            print(f"警告: 子集文件中的 {len(file_paths) - len(image_files)} 个图像路径不存在")
    else:
        # 如果没有提供子集文件，则按原来的方式获取所有图像
        image_dir = os.path.join(yolo_dataset_path, "images")
        
        for cls_name in classes:
            cls_image_dir = os.path.join(image_dir, cls_name)
            if os.path.exists(cls_image_dir):
                cls_images = glob.glob(os.path.join(cls_image_dir, "*.jpg")) + \
                            glob.glob(os.path.join(cls_image_dir, "*.jpeg")) + \
                            glob.glob(os.path.join(cls_image_dir, "*.png"))
                image_files.extend(cls_images)
        
        # 如果没有在类别子目录中找到图像，尝试直接在images目录下查找
        if not image_files:
            image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                         glob.glob(os.path.join(image_dir, "*.jpeg")) + \
                         glob.glob(os.path.join(image_dir, "*.png"))
    
    if not image_files:
        raise FileNotFoundError(f"未找到任何图像文件")
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理每个图像和对应的标签
    ann_id = 0
    for img_id, img_path in enumerate(tqdm(image_files, desc="处理图像")):
        # 获取图像信息
        img_filename = os.path.basename(img_path)
        img_name_without_ext = os.path.splitext(img_filename)[0]
        
        # 获取图像尺寸
        try:
            img = Image.open(img_path)
            img_width, img_height = img.size
        except Exception as e:
            print(f"无法打开图像 {img_path}: {e}")
            continue
        
        # 添加图像信息到COCO数据 - 使用文件名作为image_id
        coco_data["images"].append({
            "id": img_name_without_ext,  # 修改点2：使用文件名作为image_id
            "file_name": img_filename,
            "width": img_width,
            "height": img_height,
            "license": 1,
            "date_captured": ""
        })
        
        # 确定标签文件路径
        # 首先尝试根据图像路径推断标签路径
        img_dir = os.path.dirname(img_path)
        cls_name = os.path.basename(img_dir)
        
        # 尝试在类别子目录中查找
        label_found = False
        label_path = os.path.join(yolo_dataset_path, "labels", cls_name, f"{img_name_without_ext}.txt")
        if os.path.exists(label_path):
            label_found = True
        
        # 如果在类别子目录中没找到，尝试直接在labels目录下查找
        if not label_found:
            label_path = os.path.join(yolo_dataset_path, "labels", f"{img_name_without_ext}.txt")
            if not os.path.exists(label_path):
                print(f"未找到图像 {img_filename} 对应的标签文件")
                continue
        
        # 读取标签信息
        try:
            with open(label_path, 'r') as f:
                label_lines = [line.strip() for line in f.readlines() if line.strip()]
            
            # 处理每个标签
            for line in label_lines:
                parts = line.split()
                if len(parts) != 5:
                    print(f"标签格式错误: {line}")
                    continue
                
                # 解析YOLO格式的标签
                cls_id = int(parts[0])
                if cls_id >= len(classes):
                    print(f"警告: 标签中的类别ID {cls_id} 超出了类别列表范围 (0-{len(classes)-1})")
                    continue
                
                # 获取类别名称和新的类别ID
                cls_name = classes[cls_id]
                new_cls_id = class_name_to_id[cls_name]  # 修改点3：使用从1开始的类别ID
                
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # 转换为COCO格式（左上角坐标和宽高）
                x = (x_center - width / 2) * img_width
                y = (y_center - height / 2) * img_height
                w = width * img_width
                h = height * img_height
                
                # 添加标注信息到COCO数据
                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": img_name_without_ext,  # 修改点4：使用文件名作为image_id
                    "category_id": new_cls_id,  # 修改点5：使用从1开始的类别ID
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "segmentation": [],
                    "iscrowd": 0
                })
                ann_id += 1
        except Exception as e:
            print(f"处理标签文件 {label_path} 时出错: {e}")
    
    # 保存COCO格式的数据到JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成，共处理 {len(coco_data['images'])} 张图像，{len(coco_data['annotations'])} 个标注")
    print(f"COCO格式数据已保存到: {output_json_path}")

def main():
    parser = argparse.ArgumentParser(description="将YOLO格式的数据集转换为COCO格式")
    parser.add_argument("--dataset_path", type=str, default=".", help="YOLO数据集的根目录")
    parser.add_argument("--output", type=str, default="instances_val2017.json", help="输出的COCO格式JSON文件路径")
    parser.add_argument("--subset", type=str, default="val.txt", help="可选，指定包含要处理的图像列表的文本文件路径")
    args = parser.parse_args()
    
    yolo2coco(args.dataset_path, args.output, args.subset)

if __name__ == "__main__":
    main()
