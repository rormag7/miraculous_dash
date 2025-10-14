from ultralytics import YOLO
import numpy as np
import os

# Load your trained model
model = YOLO(r'C:\Users\rorym\Downloads\FALL 2025\Applied Project\Code\Model_Weights\Step_Identification\yolov8l_best.pt') 

# Run validation on the same dataset used for training
results = model.val(
                    data=r"C:\Users\rorym\Downloads\FALL 2025\Applied Project\Code\YOLO_data.yaml",
                    save_json=True,
                    save_txt=True,
                    )


def bbox_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union else 0

pred_dir = r"runs\detect\val\labels"   # YOLO saves val predictions here
gt_dir   = r"path\to\your\dataset\val\labels"

ious = []
for file in os.listdir(pred_dir):
    pred_path = os.path.join(pred_dir, file)
    gt_path   = os.path.join(gt_dir, file)
    if not os.path.exists(gt_path): 
        continue

    preds = np.loadtxt(pred_path).reshape(-1, 6)[:,1:]  # cls, x, y, w, h → xywh
    gts   = np.loadtxt(gt_path).reshape(-1, 5)[:,1:]
    for p in preds:
        px, py, pw, ph = p
        # Convert YOLO (x,y,w,h) to (x1,y1,x2,y2)
        pbox = [px-pw/2, py-ph/2, px+pw/2, py+ph/2]
        best_iou = max(bbox_iou(pbox, [gx-gw/2, gy-gh/2, gx+gw/2, gy+gh/2])
                       for gx,gy,gw,gh in gts)
        ious.append(best_iou)

mean_iou = np.mean(ious)
print(f"Mean IoU: {mean_iou:.3f}")


import matplotlib.pyplot as plt

# Split this up by class???
plt.figure(figsize=(7,4))
plt.hist(np.clip(ious, 0, 1), bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of IoU Values Across Detections')
plt.xlabel('IoU')
plt.ylabel('Number of Detections')
plt.grid(True, alpha=0.3)
plt.show()