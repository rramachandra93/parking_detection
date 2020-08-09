import cv2
import imutils
import numpy as np
from svm import SVM
from imutils.video import count_frames
from imageprocessing import ImageProcess



path = 'videos/test2.mp4'
cap = cv2.VideoCapture(path)
fps = 12
capSize = (640, 360)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter()
success = out.open('./teste5.mp4', fourcc, fps, capSize, True)

num_frames = count_frames(path)
print(num_frames)
i = 0

while cap.isOpened():
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gau = cv2.GaussianBlur(gray, (7, 7), 0)
    boxes = np.asarray([[[400.14516129, 401.74193548],
                              [318.85483871, 379.96774194],
                              [327.56451613, 336.41935484],
                              [416.11290323, 359.64516129]],

                             [[420.46774194, 359.64516129],
                              [433.53225806, 323.35483871],
                              [343.53225806, 292.87096774],
                              [337.72580645,327.70967742]],

                             [[285.46774194, 288.51612903],
                              [353.69354839, 295.77419355],
                              [369.66129032, 265.29032258],
                              [292.72580645, 252.22580645]],

                             [[272.40322581,239.16129032],
                              [342.08064516,250.77419355],
                              [352.24193548, 224.64516129],
                              [281.11290323, 210.12903226]],

                             [[249.17741935, 201.41935484],
                              [313.0483871, 205.77419355],
                              [326.11290323, 185.4516129 ],
                              [240.46774194, 172.38709677]],

                             [[225.9516129, 169.48387097],
                              [294.17741935, 169.48387097],
                              [298.53225806, 156.41935484],
                              [234.66129032, 137.5483871 ]],

                             [[207.08064516, 131.74193548],
                              [270.9516129,  143.35483871],
                              [275.30645161, 124.48387097],
                              [212.88709677, 117.22580645]],

                             [[192.56451613, 111.41935484],
                              [243.37096774, 112.87096774],
                              [249.17741935,  95.4516129 ],
                              [188.20967742,  86.74193548]],

                             [[172.24193548,  86.74193548],
                              [225.9516129,   88.19354839],
                              [234.66129032,  72.22580645],
                              [176.59677419,  69.32258065]],

                             [[156.27419355,  63.51612903],
                              [202.72580645,  64.96774194],
                              [211.43548387,  54.80645161],
                              [153.37096774, 46.09677419]],

                             [[153.37096774, 46.09677419],
                              [207.08064516, 46.09677419],
                              [207.08064516,  34.48387097],
                              [157.72580645, 21.41935484]],

                             [[199.82258065,  25.77419355],
                              [199.82258065,  15.61290323],
                              [249.17741935,  18.51612903],
                              [249.17741935,  40.29032258]],

                             [[214.33870968,  56.25806452],
                              [249.17741935,  51.90322581],
                              [252.08064516,  41.74193548],
                              [217.24193548,  37.38709677]],

                             [[224.5,         72.22580645],
                              [273.85483871,  70.77419355],
                              [275.30645161,  51.90322581],
                              [228.85483871,  51.90322581]],

                             [[241.91935484,  89.64516129],
                              [286.91935484,  99.80645161],
                              [299.98387097,  83.83870968],
                              [246.27419355,  72.22580645]],

                             [[257.88709677, 117.22580645],
                              [257.88709677,  99.80645161],
                              [301.43548387, 104.16129032],
                              [297.08064516, 118.67741935]],

                             [[297.08064516, 118.67741935],
                              [282.56451613, 139.        ],
                              [342.08064516, 147.70967742],
                              [349.33870968, 127.38709677]],

                             [[313.0483871,  179.64516129],
                              [369.66129032, 185.4516129 ],
                              [387.08064516, 163.67741935],
                              [315.9516129, 147.70967742]],

                             [[326.11290323, 207.22580645],
                              [333.37096774, 186.90322581],
                              [405.9516129,  182.5483871 ],
                              [408.85483871, 218.83870968]],

                             [[429.17741935, 263.83870968],
                              [360.9516129, 255.12903226],
                              [358.0483871,  226.09677419],
                              [430.62903226, 230.4516129 ]],

                             [[385.62903226, 294.32258065],
                              [395.79032258, 258.03225806],
                              [509.01612903, 282.70967742],
                              [501.75806452, 326.25806452]],

                             [[607.72580645, 221.74193548],
                              [614.98387097, 175.29032258],
                              [648.37096774, 181.09677419],
                              [651.27419355, 227.5483871 ]],

                             [[564.17741935, 179.64516129],
                              [565.62903226, 153.51612903],
                              [616.43548387, 153.51612903],
                              [610.62903226, 202.87096774]],

                             [[520.62903226, 147.70967742],
                              [526.43548387, 121.58064516],
                              [568.53225806, 128.83870968],
                              [567.08064516, 168.03225806]],

                             [[478.53225806, 121.58064516],
                              [493.0483871,   94.        ],
                              [527.88709677, 107.06451613],
                              [522.08064516, 150.61290323]],

                             [[450.9516129,   96.90322581],
                              [462.56451613,  76.58064516],
                              [493.0483871,   79.48387097],
                              [481.43548387, 125.93548387]],

                             [[424.82258065,  78.03225806],
                              [442.24193548,  54.80645161],
                              [468.37096774,  62.06451613],
                              [455.30645161,  96.90322581]],

                             [[394.33870968,  57.70967742],
                              [411.75806452,  30.12903226],
                              [437.88709677, 41.74193548],
                              [429.17741935,  75.12903226]],

                             [[372.56451613,  37.38709677],
                              [395.79032258, 11.25806452],
                              [413.20967742, 22.87096774],
                              [403.0483871,  49.        ]],

                             [[349.33870968, 21.41935484],
                              [365.30645161,  9.80645161],
                              [388.53225806, 12.70967742],
                              [371.11290323,  46.09677419]],

                             [[388.53225806,   9.80645161],
                              [448.0483871,    5.4516129 ],
                              [452.40322581,  17.06451613],
                              [405.9516129,   22.87096774]],

                              [[429.17741935,  27.22580645],
                              [433.53225806,  40.29032258],
                              [491.59677419,  38.83870968],
                              [485.79032258,  15.61290323]],

                             [[472.72580645,  63.51612903],
                              [442.24193548,  44.64516129],
                              [490.14516129,  38.83870968],
                              [497.40322581,  56.25806452]],

                             [[481.43548387,  60.61290323],
                              [501.75806452,  85.29032258],
                              [551.11290323,  86.74193548],
                              [539.5,  54.80645161]],

                             [[504.66129032, 80.93548387],
                              [530.79032258,105.61290323],
                              [559.82258065, 108.51612903],
                              [559.82258065,  88.19354839]],

                             [[530.79032258, 127.38709677],
                              [572.88709677, 128.83870968],
                              [565.62903226, 105.61290323],
                              [526.43548387, 105.61290323]],

                             [[584.5,        130.29032258],
                              [645.46774194, 123.03225806],
                              [654.17741935, 152.06451613],
                              [609.17741935, 150.61290323]],

                             [[632.40322581, 181.09677419],
                              [702.08064516, 191.25806452],
                              [709.33870968, 168.03225806],
                              [646.91935484, 156.41935484]],

                             [[668.69354839,  94.        ],
                              [667.24193548,  67.87096774],
                              [700.62903226,  69.32258065],
                              [712.24193548, 107.06451613]],

                             [[628.0483871,   72.22580645],
                              [632.40322581,  46.09677419],
                              [664.33870968,  51.90322581],
                              [668.69354839,  94.        ]],

                             [[594.66129032,  47.5483871 ],
                              [596.11290323,  21.41935484],
                              [629.5,          28.67741935],
                              [628.0483871,   56.25806452]],

                             [[565.62903226, 14.16129032],
                              [559.82258065,  33.03225806],
                              [594.66129032, 44.64516129],
                              [603.37096774,  21.41935484]],

                             [[533.69354839,  19.96774194],
                              [536.59677419,   1.09677419],
                              [564.17741935,   5.4516129 ],
                              [555.46774194,  28.67741935]]])
    print(boxes)

    img_resize = ImageProcess.get_rectangle(gau, boxes)
    feature = ImageProcess().extract_features(img_resize)

    score = SVM().predict(feature)
    for k in range(43):
        if score[k] == 1:
            cv2.polylines(frame, np.int32([boxes[k]]), True, (0, 0, 255), 2)
        else:
            cv2.polylines(frame, np.int32([boxes[k]]), True, (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    key = cv2.waitKey(220) & 0xFF
    if key == ord("q"):
        break