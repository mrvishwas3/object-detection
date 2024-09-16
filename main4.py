import cv2
from ultralytics import YOLO
import pandas as pd
from tracker import Tracker
import numpy as np
import cvzone

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

model = YOLO("best.pt")

cap = cv2.VideoCapture('chicks.mp4')

# Define the output aspect ratio and frame dimensions
frame_height = 600
frame_width = int(frame_height * (9 / 16))  # Aspect ratio 9:16

# Create a video writer to save the output with the same aspect ratio
output_file = 'output_9_16.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter(output_file, fourcc, 30.0, (frame_width, frame_height))

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count = 0
tracker = Tracker()
chickscount = []
cy1 = int(frame_height * (1 / 4))  # Adjusted for the 9:16 frame height
offset = 6

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    # Resize the frame to 9:16 aspect ratio
    frame_resized = cv2.resize(frame, (frame_width, frame_height))
    
    results = model(frame_resized)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    bbox_list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        if d < len(class_list):
            c = class_list[d]
        else:
            print(f'Warning: Index {d} is out of range for class list.')
            continue
        
        bbox_list.append([x1, y1, x2 - x1, y2 - y1])  # Convert to (x, y, w, h)
        
    tracked_bboxes = tracker.update(bbox_list)
    for bbox in tracked_bboxes:
        x, y, w, h, obj_id = bbox
        cx = (x + x + w) // 2
        cy = (y + y + h) // 2
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame_resized, (cx, cy), 4, (0, 0, 255), -1)
            cvzone.putTextRect(frame_resized, f'{obj_id}', (x, y), 1, 1)
            if obj_id not in chickscount:
                chickscount.append(obj_id)
    
    counting = len(chickscount)
    cvzone.putTextRect(frame_resized, f'{counting}', (50, 60), 2, 2)
    cv2.line(frame_resized, (5, cy1), (frame_width - 5, cy1), (255, 0, 255), 2)
    
    # Write the frame to the output video file
    out.write(frame_resized)
    
    # Show the frame in a window
    cv2.imshow("RGB", frame_resized)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
out.release()
cv2.destroyAllWindows()
