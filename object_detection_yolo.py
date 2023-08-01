```python
import cv2
from darknet import Darknet

# Load YOLO
def load_yolo():
    net = Darknet("yolo.cfg")
    net.load_weights("yolo.weights")
    classes = []
    with open("yolo.classes", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes

# Perform object detection
def detect_objects(img, net, classes):
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    return boxes, confidences, class_ids

# Draw bounding boxes on the detected objects
def draw_bounding_box(img, boxes, confidences, class_ids, classes):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
    cv2.imshow("Image", img)

# Main function
def main():
    image = cv2.imread("input.jpg")
    net, classes = load_yolo()
    boxes, confidences, class_ids = detect_objects(image, net, classes)
    draw_bounding_box(image, boxes, confidences, class_ids, classes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```