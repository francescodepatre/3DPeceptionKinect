import freenect
import cv2
import numpy as np
import time




def calibration(angle):
    ctx = freenect.init()
    device = freenect.open_device(ctx, 0)
    freenect.set_tilt_degs(device, angle)
    time.sleep(3)
    freenect.close_device(device)
    freenect.shutdown(ctx)
    
calibration(+27.00)
time.sleep(3)
calibration(-27.00)
time.sleep(3)
calibration(0.00)


#Cambiare con il path del file con le classi
with open('/home/francesco-de-patre/darknet/coco.names','r') as f:
    classes = [line.strip() for line in f.readlines()]

#Cambiare con il path dei file di configurazione e dei pesi, in questo caso i file di configurazione sono stati modificati per riconoscere i volti
net = cv2.dnn.readNet('/home/francesco-de-patre/darknet/yolov3-custom_last.weights','/home/francesco-de-patre/darknet/yolov3-custom.cfg')

cv2.ocl.setUseOpenCL(True)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while True:

    rgb_image = freenect.sync_get_video()[0]
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    depth_map = freenect.sync_get_depth()[0]

    height, width, _ = rgb_image.shape

    blob = cv2.dnn.blobFromImage(rgb_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    
    outputs = net.forward(output_layers)


    class_ids = []
    confidences = []
    boxes = []

    
    for output in outputs:
        
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices:
            box = boxes[i]
            x, y, w, h = box
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            person_center_x = x + w // 2
            person_center_y = y + h // 2

            if 0 <= person_center_x < width and 0 <= person_center_y < height:
                distance = depth_map[person_center_y, person_center_x]  
                distance_meters = distance / 1000.0  
        
            cv2.rectangle(rgb_image, (x,y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(rgb_image, f"{label} {confidence:.2f}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(rgb_image, f"disance: {distance_meters:.2f}m", (x, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Output - Press q to exit', rgb_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
     break

cv2.destroyAllWindows()

