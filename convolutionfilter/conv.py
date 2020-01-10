from math import ceil
from multiprocessing import Manager
from typing import List, Tuple

import numpy as np
from PIL import Image

from convolutionfilter.worker import _WorkersManager, _ConvWorker, _WorkerResult


class Conv:
    EXTENSION = '.ppm'
    RESULT_FILE_NAME = f'result{EXTENSION}'

    def __init__(self, img: np.ndarray, matrix: np.ndarray, number_of_workers: int, iterations: int) -> None:
        self._img = img
        self._matrix = matrix
        self._number_of_workers = number_of_workers
        self._iterations = iterations

        self._new_img = None
        self._manager = _WorkersManager()
        self._lock_manager = Manager()
        self._workers: List[Tuple[_ConvWorker, _WorkerResult]] = []
        self._chunk = int(ceil(len(self._img) / self._number_of_workers))

    def apply(self):
        self._start_workers()
        self._join_workers()

    def _start_workers(self):
        processed_rows = 0
        for n in range(0, self._number_of_workers-1):
            self._start_worker(n, processed_rows, self._chunk)
            processed_rows += self._chunk
        self._start_worker(self._number_of_workers-1, processed_rows, len(self._img) - processed_rows)

    def _start_worker(self, n: int, start: int, chunk: int):
        print(f'worker {n} {start}-{start + self._chunk}')
        # noinspection PyUnresolvedReferences
        result: _WorkerResult = self._manager.result(chunk, len(self._img[0]))
        worker = _ConvWorker(n, self._iterations, self._img[start:start + chunk], self._matrix, result, self._lock_manager.Lock(), self._lock_manager.Lock())
        worker.start()
        self._workers.append((worker, result))

    def _join_workers(self):
        for worker in self._workers:
            worker[0].join()
            if self._new_img is None:
                self._new_img = np.concatenate((worker[1].get(),))
            else:
                self._new_img = np.concatenate((self._new_img, worker[1].get()))

    def save_result(self):
        Image.fromarray(self._new_img).save(Conv.RESULT_FILE_NAME)
