from typing import List, Tuple

import numpy as np
from PIL import Image

from convolutionfilter.worker import _WorkersManager, _ConvWorker, _WorkerResult


class Conv:
    EXTENSION = '.jpg'
    RESULT_FILE_NAME = f'result{EXTENSION}'

    def __init__(self, img: np.ndarray, matrix: List[List[int]], number_of_workers: int) -> None:
        self._img = img
        self._matrix = matrix
        self._number_of_workers = number_of_workers

        self._new_img = None
        self._manager = _WorkersManager()
        self._workers: List[Tuple[_ConvWorker, _WorkerResult]] = []
        self._chunk = int(len(self._img) / self._number_of_workers)

    def apply(self):
        self._start_workers()
        self._join_workers()

    def _start_workers(self):
        for n in range(0, self._number_of_workers):
            self._start_worker(n)

    def _start_worker(self, n: int):
        start = n * self._chunk + 1
        print(f'worker {n} {start}-{start + self._chunk}')
        # noinspection PyUnresolvedReferences
        result: _WorkerResult = self._manager.result(self._chunk, len(self._img[0]))
        worker = _ConvWorker(
            n,
            self._img[start:start + self._chunk],
            self._matrix,
            result
        )
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
