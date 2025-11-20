#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把 DUT Detection 的 VOC XML 标注 转成 YOLO txt 格式。

输入目录结构（ROOT_IN）：
DUT/
  train/
    img/*.jpg
    xml/*.xml
  val/
    img/*.jpg
    xml/*.xml
  test/
    img/*.jpg
    xml/*.xml

输出目录结构（ROOT_OUT）：
DUT_YOLO/
  images/train/*.jpg
  images/val/*.jpg
  images/test/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  labels/test/*.txt
"""

import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

# ======== 根据你自己的路径改这里即可 ========
ROOT_IN  = Path("D:\gongchuang\DUT")          # 你的 DUT 根目录
ROOT_OUT = Path("D:\gongchuang\DUT_YOLO")     # 输出 YOLO 数据集目录
SPLITS   = ["train", "val", "test"]
CLASS_MAP = {"UAV": 0}  # 只有一个类 UAV，映射到 0
# =========================================


def convert_xml_to_yolo(xml_path: Path, out_txt_path: Path):
    """单个 VOC XML -> YOLO txt"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    w = float(size.find("width").text)
    h = float(size.find("height").text)

    lines = []

    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS_MAP:
            continue
        class_id = CLASS_MAP[name]

        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        # 防止越界
        xmin = max(0.0, min(xmin, w - 1))
        xmax = max(0.0, min(xmax, w - 1))
        ymin = max(0.0, min(ymin, h - 1))
        ymax = max(0.0, min(ymax, h - 1))

        bw = xmax - xmin
        bh = ymax - ymin
        if bw <= 0 or bh <= 0:
            continue

        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0

        # 归一化到 0~1
        cx /= w
        cy /= h
        bw /= w
        bh /= h

        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with out_txt_path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    ROOT_OUT.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        img_dir = ROOT_IN / split / "img"
        xml_dir = ROOT_IN / split / "xml"

        if not img_dir.exists() or not xml_dir.exists():
            print(f"[WARN] 子目录缺失，跳过: {split}")
            continue

        out_img_dir = ROOT_OUT / "images" / split
        out_lbl_dir = ROOT_OUT / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        xml_files = sorted(xml_dir.glob("*.xml"))
        print(f"[{split}] 找到 {len(xml_files)} 个 XML 标注文件")

        for xml_path in xml_files:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            filename_node = root.find("filename")
            if filename_node is not None:
                img_name = filename_node.text.strip()
            else:
                img_name = xml_path.stem + ".jpg"

            img_path = img_dir / img_name
            if not img_path.exists():
                print(f"[WARN] 找不到图片: {img_path}")
                continue

            # 拷贝图片到 YOLO 的 images 目录
            dst_img_path = out_img_dir / img_name
            if not dst_img_path.exists():
                shutil.copy2(img_path, dst_img_path)

            # 生成对应的 YOLO 标签
            out_txt_path = out_lbl_dir / (img_path.stem + ".txt")
            convert_xml_to_yolo(xml_path, out_txt_path)

        print(f"[{split}] 转换完成，images -> {out_img_dir}, labels -> {out_lbl_dir}")

    print("全部完成 ✅")


if __name__ == "__main__":
    main()
