import cv2
import argparse
import numpy as np
import time
import datetime

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True,
                    help = 'path to yolo config file')
    ap.add_argument('-w', '--weights', required=True,
                    help = 'path to yolo pre-trained weights')
    ap.add_argument('-cl', '--classes', required=True,
                    help = 'path to text file containing class names')
    ap.add_argument('-i', '--image', required=False,
                    help = 'path to input image')
    ap.add_argument('-v', '--video', required=False,
                    help = 'path to input video')
    args = ap.parse_args()

    classes = None

    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))


    def image_detect(image):
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        
        net = cv2.dnn.readNet(args.weights, args.config)

        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4


        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])


        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), classes, COLORS)

        return image


    def video_detect(cap):

        # get the information of cap
        codec = cv2.VideoWriter_fourcc(*'MJPG') #int(cap.get(cv2.CAP_PROP_FOURCC))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # output stream
        output = cv2.VideoWriter('./video_output.avi', codec, fps, (frameWidth,frameHeight))
        frameIndex = cap.get(cv2.CAP_PROP_POS_FRAMES)
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        while True:
            ret, frame = cap.read()
            if ret:
                result = image_detect(frame)
                cv2.imshow("object detection", result)
                cv2.waitKey(1)
                
                frameIndex = cap.get(cv2.CAP_PROP_POS_FRAMES)
                print("frameIndex")
                print(frameIndex)
                output.write(frame)
            else:
                break
                
        cap.release()
        output.release()
        cv2.destroyAllWindows()


    # might start from here
    if args.image:
        image = cv2.imread(args.image)
        image = image_detect(image)
        cv2.imshow("object detection", image)
        cv2.waitKey()
            
        cv2.imwrite("object-detection.jpg", image)
        cv2.destroyAllWindows()

    if args.video:
        cap = cv2.VideoCapture(args.video)
        video_detect(cap)


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes, COLORS):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    
if __name__ == '__main__':
    main()