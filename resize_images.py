import os
import cv2

def resize_images(input_dir, output_dir, target_size=(128, 128)):
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        save_path = os.path.join(output_dir, class_name)
        os.makedirs(save_path, exist_ok=True)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            # ✅ Skip folders and non-image files
            if not os.path.isfile(img_path):
                continue

            img = cv2.imread(img_path)
            if img is not None:
                resized = cv2.resize(img, target_size)
                cv2.imwrite(os.path.join(save_path, img_name), resized)

# ✅ Run this once for both train and val folders
resize_images("data/train", "data_resized/train")
resize_images("data/val", "data_resized/val")