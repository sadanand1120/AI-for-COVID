import argparse
import threading
from queue import Queue

import cv2
import imutils
from imutils.video import FPS
from imutils.video import FileVideoStream

from scripts.all_behaviours import AllBehaviours

ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--input1", required=True, type=str)
ap.add_argument("-i2", "--input2", required=True, type=str)
args = vars(ap.parse_args())


def preprocess(frame):
    return imutils.resize(frame, width=500)


def overlayAndShow(frame, outputs):
    # Unpack all behaviours outputs
    for cur_behav_out in outputs:
        behav_name = cur_behav_out[0]
        cur_behav_out = cur_behav_out[1:]
        if behav_name == "Face-mask":
            for box in cur_behav_out:
                (x1, y1, x2, y2, total_conf, cls_pred) = box  # cls_pred == 0 means MASK
                cls_pred = int(cls_pred)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255 * (1 - cls_pred), 255 * cls_pred), 2)
    return frame


class CustomBehaviourThread(threading.Thread):
    def __init__(self, inQ, indexB, camind, obj):
        super(CustomBehaviourThread, self).__init__()
        self.inQ = inQ
        self.indexB = indexB
        self.camind = camind
        self.obj = obj

    def run(self):
        global outputQs, NUM_VIDS, NUM_BEHAVIOURS, BEHAVIOURS_NAMES
        while True:
            if not self.inQ.empty():
                frame = list(self.inQ.queue)[0]  # basically its just inQ.get() but without removing!
                if frame is None:
                    break
                frame = preprocess(frame)
                if BEHAVIOURS_NAMES[self.indexB] == "Face-mask":
                    out = self.obj.faceMaskDetector(frame)
                    # Just for testing, uncomment below line, and comment out above line, for having fixed random
                    # output!
                    # out = ["Face-mask", (1, 1, 20, 20, 0.9, 1), (50, 50, 100, 100, 0.8, 0)]
                    outputQs[self.camind][self.indexB].put(out)


# Create a subclass of the threading class. This creates a thread for each camera, and overlays our two behaviours
# onto it. And then outputs the image.

class CustomMainThread(threading.Thread):
    def __init__(self, src, ind):
        super(CustomMainThread, self).__init__()
        self.src = src  # the input camera/video source link
        self.fvs = FileVideoStream(self.src, queue_size=64).start()
        self.ind = ind
        self.inputQ = self.fvs.Q
        self.obj = AllBehaviours()

    def run(self):
        global outputQs, NUM_VIDS, NUM_BEHAVIOURS, BEHAVIOURS_NAMES
        BehavList = []
        for i in range(NUM_BEHAVIOURS):
            t = CustomBehaviourThread(self.inputQ, i, self.ind, self.obj)
            t.daemon = True
            t.start()
            BehavList.append(t)
        fpstot = FPS().start()
        while self.fvs.more():
            # take input
            frame = self.fvs.read()
            if frame is None:
                break
            frame = preprocess(frame)
            outs = []
            for i in range(NUM_BEHAVIOURS):
                out = outputQs[self.ind][i].get()
                outs.append(out)
            frame = overlayAndShow(frame, outs)
            cv2.imshow(f"cam {self.ind}", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            fpstot.update()
        self.fvs.stop()
        fpstot.stop()
        print(f"Fps is {round(fpstot.fps(), 2)} for video {self.ind}")
        cv2.destroyWindow(f"cam {self.ind}")


outputQs = []
NUM_VIDS = 2
NUM_BEHAVIOURS = 1
BEHAVIOURS_NAMES = ["Face-mask"]
for _ in range(NUM_VIDS):
    Blist = []
    for _ in range(NUM_BEHAVIOURS):
        q = Queue()
        Blist.append(q)
    outputQs.append(Blist)

src1 = args["input1"]
src2 = args["input2"]

t1 = CustomMainThread(src1, 0)
t2 = CustomMainThread(src2, 1)
t1.start()
t2.start()
t1.join()
t2.join()
