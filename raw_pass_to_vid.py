import numpy as np
import cv2

trial_file = "S145_W1.npz" 
 
data = np.load(trial_file)
trial_frames = data['arr_0'][451:885]
#trial_frames = trial_frames.astype(np.uint8)

# Create a VideoWriter
height, width, frames = trial_frames.shape
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height), isColor=True)

# Write each frame
for i in range(frames):
    frame = trial_frames[:, :, i]
    out.write(frame)

out.release()
print("Video saved as output.mp4")