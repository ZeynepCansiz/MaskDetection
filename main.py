import cv2
import numpy as np
import math

face_cascade = cv2.CascadeClassifier(r"Classifier/haarcascade_frontalface_alt.xml")

font = cv2.FONT_HERSHEY_SIMPLEX

net = cv2.dnn.readNet('yolov3_custom_last.weights', 'yolov3_custom.cfg')

classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

capture = cv2.VideoCapture(r"video/son.MOV")
colors = np.random.uniform(0, 255, size=(100, 3))
ber = 0
bak = 5

while capture.isOpened():
    _,img = capture.read()
    height, width, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray,1.05,3)
    l = [] ##koordinatların ekleneceği liste
    lf = [] ##sıra sayısını tutan liste
    i = 1
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y + 20), font, 1, (255, 255, 255), 2)
            """metin = "Maske Gördüm"
            text_to_speech(metin)"""



    for (x,y,w,h) in faces:
        s = str(i)
        if len(indexes) > 0:
            pass
        else:
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 3)
            cv2.putText(img,s,(x,y),font,1,color,2)

        i += 1
        l = []
        l.append(x)
        l.append(y)
        lf.append(l)

        print(l)

    print(lf)

    close_person = ""
    for i in range(len(lf)):
        for j in range(i+1, len(lf)):
            d = math.sqrt(( (lf[j][1] - lf[i][1])**2) +((lf[j][0]-lf[i][0])**2) )
            print("P", i+1, "- P", j+1, "=", d)

            if d<400:
                close_person += "Person " + str(i+1) + " and Person " + str(j+1) + " ; "
                cv2.line(img, (lf[i][0],lf[i][1]), (lf[j][0],lf[j][1]), (0,0,255), 2)

    close_person += "are not following social distancing"
    print(close_person)

    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
