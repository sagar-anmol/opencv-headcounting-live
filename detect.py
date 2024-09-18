import numpy as np
import cv2


# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)
whT = 320  # Width and height target for YOLO input image size
confThreshold = 0.5  # Confidence threshold for filtering weak detections
nmsThreshold = 0.3  # Non-maximum suppression threshold

# Load class names from COCO dataset
classesfile = 'obj.names'
classNames = []

with open(classesfile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load YOLOv3 configuration and pre-trained weights
modelConfig = 'the.cfg'
modelWeights = 'the.weights'
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# We want to detect only the first class from coco.names (usually "person")
targetClassId = 0  # Index 0 corresponds to the first object in coco.names

def findObject(outputs, img):
    hT, wT, cT = img.shape
    bbox = []  # Bounding box coordinates
    classIds = []  # Detected class IDs
    confs = []  # Confidence scores
    
    # Iterate over each detection in the network output
    for output in outputs:
        for det in output:
            scores = det[5:]  # Skip the first 5 elements (bbox, center, etc.)
            classId = np.argmax(scores)  # Get class ID with the highest score
            confidence = scores[classId]  # Confidence of the detected object
            
            # Only process if the detected object is the target class (first object, like "person")
            if classId == targetClassId and confidence > confThreshold:
                # Calculate bounding box coordinates
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    
    # Apply non-maximum suppression to remove redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    
    # If there are any valid indices, draw boxes and labels on the image
    if len(indices) > 0:
        for i in indices.flatten():  # Flatten in case it's a 2D array
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Print the number of detected objects (only target object)
        print(f'{classNames[targetClassId].upper()} detected: {len(indices)}')

# Main loop to capture video frames and detect objects
while True:
    success, img = cap.read()  # Capture frame from webcam
    if not success:
        break
    
    # Create a blob from the image (YOLO input format)
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    
    # Get YOLO layer names
    layernames = net.getLayerNames()
    outputNames = [layernames[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Forward pass to get outputs from the network
    outputs = net.forward(outputNames)
    
    # Find objects and draw bounding boxes
    findObject(outputs, img)

    # Display the frame with detections
    cv2.imshow('Image', img)
    

    
    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
