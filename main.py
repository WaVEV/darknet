from pyDarknet.libpydarknet import DarknetObjectDetector
import cv2
import sys


specs = "cfg/yolo.cfg"
weights = "yolo.weights"

_net = DarknetObjectDetector(specs, weights)

to_string = 0.0
fordward = 0.0

def detect(im):
    """
    Detect faces in the given image

    :param img: Image for detect faces.
    :type specs: numpy.ndarray
    :return: List with bounding boxes (dlib rectangle).
    :rtype: list
    """
    global to_string, fordward
    img = cv2.resize(im, (448, 448), interpolation = cv2.INTER_CUBIC)

    img_t = img.transpose([2, 0, 1])
    img_str = img_t.tostring()

    dets = _net.detect_object(img_str, img_t.shape[2],    # img h
                                   img_t.shape[1],    # img w
                                   img_t.shape[0])
    bboxs = []
    scale_y = im.shape[0] / 448.0;
    scale_x = im.shape[1] / 448.0;
    print dets
    for bb in dets:
        print bb.confidence, bb.cls
        if bb.confidence > 0.4:
            bboxs.append(map(int, [bb.left * scale_x, bb.top * scale_y, bb.right * scale_x, bb.bottom * scale_y]))
    return bboxs

im = cv2.imread(sys.argv[1])
bboxs = detect(im)
dest_path = "salida.jpg"

max_area = -1
max_bbox = None
for bb in bboxs:
    area = (bb[2] - bb[0]) * (bb[3] - bb[1])
    if max_area < area:
        max_area = area
        max_bbox = bb


for bb in bboxs:
    area = (bb[2] - bb[0]) * (bb[3] - bb[1])
    if max_area == area:
        cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), (0,0,255))
    else:
        cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), (0,255,0))

bb = max_bbox

cv2.imwrite("paint.jpg", im)
cv2.imwrite(dest_path, im[bb[1]:bb[3], bb[0]:bb[2]])


