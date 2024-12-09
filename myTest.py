from ultralytics import YOLO
import cv2 as cv

model = YOLO("/home/pnnp/Downloads/best.pt")
capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    results = model(frame)

    if len(results[0].boxes) > 0:
        bbx = results[0].boxes.cpu().numpy().xywh
        if bbx.shape[0] > 0:
            x, y, w, h = bbx[0]
            cv.rectangle(results[0].orig_img, (int(x-w/2), int(y-h/2)),
                         (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)
        else:
            pass
    else:
        pass
    cv.imshow('Test', results[0].orig_img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
