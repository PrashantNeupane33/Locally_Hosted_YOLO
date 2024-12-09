from ultralytics import YOLO
import cv2 as cv
import depthai as dai
import numpy as np

model = YOLO("/home/pnnp/Downloads/best.pt")


def rescale(frame, scale):
    w = int(frame.shape[1]*scale)
    h = int(frame.shape[0]*scale)
    return cv.resize(frame, (w, h), interpolation=cv.INTER_AREA)


def recMask(x, y, w, h, imgH, imgW):
    mask = np.zeros((imgH, imgW), dtype=np.uint8)
    cv.rectangle(mask, (int(x-w/2), int(y-h/2)),
                 (int(x+w/2), int(y+h/2)), 255, -1)

    return mask


# Creating a Pipeline
pipeline = dai.Pipeline()

# Defining Nodes
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")

# Nodes Properties
camRgb.setPreviewSize(600, 600)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Linking Nodes
camRgb.preview.link(xoutRgb.input)

# Starting the pipeline
with dai.Device(pipeline) as device:
    while True:
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        inRgb = qRgb.get()
        frame = rescale(inRgb.getCvFrame(), 1.0)

        results = model(frame)

        if len(results[0].boxes) > 0:
            bbx = results[0].boxes.cpu().numpy().xywh
            imgSize = results[0].orig_shape
            imgH, imgW = imgSize
            if bbx.shape[0] > 0:
                x, y, w, h = bbx[0]
                cv.rectangle(results[0].orig_img, (int(x-w/2), int(y-h/2)),
                             (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)
                mask = recMask(x, y, w, h, imgH, imgW)
                mask=cv.merge([mask,mask,mask])
                maskedIMG = cv.bitwise_and(
                    results[0].orig_img, mask)
                cv.imshow('Masked', maskedIMG)
            else:
                pass
        else:
            pass

        cv.waitKey(3)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()
