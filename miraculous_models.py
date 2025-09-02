from ultralytics import YOLO

# Load your best trained model
step_identification_l = YOLO('runs/detect/train/weights/best.pt')  # or wherever your model is saved

results = model.predict(
    source='path/to/image_or_folder',  # could be a file, folder, list, or even a NumPy array
    imgsz=736,                         # match training image size (if resized during training)
    conf=0.25,                         # confidence threshold (optional)
    save=True                          # save output image(s) with boxes
)

import matplotlib.pyplot as plt

# Plot the first image's predictions
results[0].show()  # opens window
# or
plt.imshow(results[0].plot())  # returns an image with boxes drawn
plt.axis('off')
plt.show()


for r in results:
    for box in r.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        print(f"Class: {class_id}, Confidence: {confidence:.2f}, Box: {xyxy}")
        
        
        
        
        
#Function to convert the heatmap into an image and save for YOLO model
def save_heatmap_to_image(heatmap, file_name, image_save_dir, cmap):
    height, width = heatmap.shape
    
    images_path = f'{image_save_dir}{file_name}.png' 
    
    dpi = 100  # Set high enough to avoid scaling
    figsize = (width / dpi, height / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])  # fill the whole figure
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(heatmap, cmap=cmap, aspect='auto')
    fig.savefig(images_path, dpi=dpi)
    plt.close(fig)