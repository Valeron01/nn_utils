import numpy as np
import glob
import os
import time
import threading
from queue import Queue
import cv2


def crop_to_n(im, n=16):
    h = round(len(im) / n)
    w = round(len(im[0]) / n)

    return cv2.resize(im, (w*16, h*16))


class Worker(threading.Thread):
    def __init__(self, queue: Queue, path, count=-1):

        self.queue = queue
        self.counter = 0
        self.path = path
        self.image_paths = glob.glob(os.path.join(path, '*.*'))[:count]
        np.random.shuffle(self.image_paths)

        self.__is_alife = True

        super().__init__(daemon=True)

    def run(self):
        while self.__is_alife:
            if not self.queue.full():
                if self.counter >= len(self.image_paths):
                    self.counter = 0
                    np.random.shuffle(self.image_paths)

                image = cv2.imread(self.image_paths[self.counter])
                
                self.queue.put(image)
                self.counter += 1
            time.sleep(0.001)

    def kill(self):
        self.__is_alife = False


class DataGenerator:
    def __init__(self, path, size=None, batch_size=1, crop_to=1,
                 pre_fun=lambda x: x, post_fun=lambda x: x):
        
        self.batch_size = batch_size

        self.queue = Queue(100)
        self.worker = Worker(self.queue, path, size=size)
        self.worker.start()
        self.size = size
        self.crop_to = crop_to

        self.post_fun = post_fun
        self.pre_fun = pre_fun


    def get_random_images(self):
        assert self.batch_size > 0

        inputs = []
        outputs = []

        for i in range(self.batch_size):
            input = self.queue.get()
            input = self.pre_fun(input)
            
            inputs.append(input)

            outputs.append(self.post_fun(input))

        return np.float32(inputs), np.float32(outputs)

    def kill(self):
        self.worker.kill()
        self.queue = None

    def queue_size(self):
        return self.queue.qsize()
