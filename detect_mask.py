from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load models
faceNet = cv2.dnn.readNet(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

maskNet = load_model("mask_detector.keras")

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (600, 400))
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * \
                  np.array([w, h, w, h])

            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]

            if face.size == 0:
                continue

            face = cv2.resize(face, (224, 224))
            face = face / 255.0
            face = np.reshape(face, (1, 224, 224, 3))

            (mask, noMask) = maskNet.predict(face)[0]

            label = "Mask" if mask > noMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            text = f"{label}: {max(mask, noMask)*100:.2f}%"

            cv2.putText(frame, text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), color, 2)

    cv2.imshow("Mask Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()