import os
import random
import cv2
from ultralytics import YOLO
from tracker import Tracker

# give saved video path here to run model on
video_path = os.path.join('.', 'data', 'video_2.mp4')

# give path where to save video
video_out_path = os.path.join('.', 'out_2.mp4')

# this line is if you want to run this model on saved video
# cap = cv2.VideoCapture(video_path)

# this line is if you want to do real time tracking using camera.
cap = cv2.VideoCapture(0)

# this line read frame from captured video or real time tacking
ret, frame = cap.read()

# this line set the ouptut video features like type, FPS, height and width
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# using the yolvo v8 model which is train on 80 objects, list is given in coco_classes.txt file
model = YOLO("YOLO_models/yolov8n.pt")

# tracker is used for deepsort tracking ( creating boxes in every frame )
tracker = Tracker()

# generate random 10 colors
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5

# running loop until the last frame of the video or real time tracking
while ret:

    # giving each frame to the model and get its results on it
    # so in this yolvo v8 detect objects name,
    results = model(frame)

        # Iterate over the results obtained from the YOLO model
    for result in results:
        # Initialize an empty list to store detections
        detections = []
        # Initialize an object_id variable to 0
        object_id = 0
        # Iterate over the bounding boxes of detected objects
        for r in result.boxes.data.tolist():
            # Unpack the bounding box coordinates, score, and class_id from the result
            x1, y1, x2, y2, score, class_id = r

            # Convert the bounding box coordinates and class_id to integers
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)

            # Assign the class_id to the object_id
            object_id = class_id

            # If the score is greater than the detection threshold, append the detection to the detections list
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])

            # Update the tracker with the current frame and detections
            tracker.update(frame, detections)

            # find object using coco sheet

            # Load class names from a file
            with open("coco_classes.txt", "r") as file:
                class_names = [line.strip() for line in file]

            # Get the name of the detected object using the object_id as index
            object_name = class_names[object_id]

            # Iterate over the tracks in the tracker
            for track in tracker.tracks:
                # Get the bounding box of the track
                bbox = track.bbox
                # Unpack the bounding box coordinates
                x1, y1, x2, y2 = bbox
                # Get the track id
                track_id = track.track_id

                # Select a color for the bounding box based on the track id
                selected_color = colors[track_id % len(colors)]
                # Draw a rectangle around the detected object in the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), selected_color , 3)
                # Add a label with the name of the detected object to the frame
                cv2.putText(frame, f"{object_name}", (int(x1), int(y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, selected_color, 2)

    # Write the frame to the output video
    # cap_out.write(frame)

    # Show the frame
    cv2.imshow('frame', frame)

    # Wait for 30 milliseconds
    cv2.waitKey(30)

    # Read the next frame
    ret, frame = cap.read()

# Release the video capture and video writer
cap.release()

cap_out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
