import os
import xml.etree.ElementTree as ET
import glob
import shutil
import random

IMG_WIDTH = 960.0
IMG_HEIGHT = 540.0

CLASS = {
    "car" : 0,
    "bus" : 1,
    "van" : 2,
    "others" : 3
}

def parse_xml_to_yolo(xml_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for frame in root.findall('.//frame'):
        frame_num = int(frame.attrib['num'])
        txt_filename = f"img{frame_num:05d}.txt"
        txt_filepath = os.path.join(output_path, txt_filename)

        with open(txt_filepath, "w") as f:
            target_list = frame.find('target_list')
            if target_list is None:
                continue
            for target in target_list.findall('target'):
                vehicle_type = target.find('attribute').attrib['vehicle_type']
                if vehicle_type not in CLASS.keys():
                    continue
                class_id = CLASS[vehicle_type]
                box = target.find('box')
                left = float(box.attrib['left'])
                top = float(box.attrib['top'])
                width = float(box.attrib['width'])
                height = float(box.attrib['height'])

                x_center = left + width / 2.0
                y_center = top + height / 2.0

                x_center_norm = x_center / IMG_WIDTH
                y_center_norm = y_center / IMG_HEIGHT
                width_norm = width / IMG_WIDTH
                height_norm = height / IMG_HEIGHT

                x_center_norm = min(max(x_center_norm, 0), 1)
                y_center_norm = min(max(y_center_norm, 0), 1)
                width_norm = min(max(width_norm, 0), 1)
                height_norm = min(max(height_norm, 0), 1)
                f.write(f"{CLASS[vehicle_type]} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
    print(f"DONE {xml_path}")

def build_yolo_dataset(raw_imgs_dir, raw_xml_dir, output_yolo_dir, split_ratio):
    dirs = [
        os.path.join(output_yolo_dir, "images", "train"),
        os.path.join(output_yolo_dir, "images", "val"),
        os.path.join(output_yolo_dir, "labels", "train"),
        os.path.join(output_yolo_dir, "labels", "val"),
    ]

    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

    xml_files = glob.glob(os.path.join(raw_xml_dir, "*.xml"))
    random.seed(42)
    random.shuffle(xml_files)

    train_size = int(len(xml_files) * split_ratio)

    subsets = {
        "train" : xml_files[:train_size],
        "val": xml_files[train_size:],
    }

    for subset_name, subset_files in subsets.items():
        for xml_file in subset_files:
            video_name = os.path.splitext(os.path.basename(xml_file))[0]

            src_img_folder = os.path.join(raw_imgs_dir, video_name)
            dst_img_folder = os.path.join(output_yolo_dir, "images", subset_name, video_name)
            dst_label_folder = os.path.join(output_yolo_dir, "labels", subset_name, video_name)

            if os.path.exists(src_img_folder):
                if not os.path.exists(dst_img_folder):
                    shutil.copytree(src_img_folder, dst_img_folder)

                parse_xml_to_yolo(xml_file, dst_label_folder)

    yaml_path = os.path.join(output_yolo_dir, "data.yaml")
    abs_dataset_dir = os.path.abspath(output_yolo_dir).replace('\\', '/')

    yaml_content = f"""path: {abs_dataset_dir}
train: images/train
val: images/val
nc: 4
names: ['car', 'bus', 'van', 'others']
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print("DONE")

if __name__ == "__main__":
    raw_img_dir = r"C:\Users\Asus\Desktop\DL\dts\DETRAC-Images\DETRAC-Images"
    raw_xml_dir = r"C:\Users\Asus\Desktop\DL\dts\DETRAC-Train-Annotations-XML\DETRAC-Train-Annotations-XML"
    output_yolo_dataset = r"C:\Users\Asus\Desktop\DL\DETRAC"

    build_yolo_dataset(raw_img_dir, raw_xml_dir, output_yolo_dataset, split_ratio=0.8)



