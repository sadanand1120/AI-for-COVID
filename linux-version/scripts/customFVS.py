# import the necessary packages
from threading import Thread
import cv2
import time


class FileVideoStream:
    def __init__(self, path, transform=None, queue_size=16, num_queues=1, queue_type="Q"):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.transform = transform
        self.num_queues = num_queues
        self.queue_type = queue_type
        self.qlist = []
        if self.queue_type == "mQ":
            from multiprocessing import Queue
        else:
            from queue import Queue
        for _ in range(self.num_queues):
            q = Queue(maxsize=queue_size)
            self.qlist.append(q)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                break

            if not self.qlist[0].full():
                (grabbed, frame) = self.stream.read()

                if not grabbed:
                    self.stopped = True

                if self.transform:
                    frame = self.transform(frame)

                for i in range(self.num_queues):
                    self.qlist[i].put(frame)
            else:
                time.sleep(0.1)

        self.stream.release()

    def read(self):
        return self.qlist[0].get()

    def running(self):
        return self.more() or not self.stopped

    def more(self):
        tries = 0
        while self.qlist[0].qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.qlist[0].qsize() > 0

    def stop(self):
        self.stopped = True
        self.thread.join()
