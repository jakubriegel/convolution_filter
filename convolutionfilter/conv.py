from math import ceil
from multiprocessing import Manager, Lock
from typing import List

import numpy as np
from PIL import Image

from convolutionfilter.worker import _WorkersManager, _ConvWorker, _WorkerResult, _WorkerRow


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
        self._workers: List[_ConvWorker] = []
        self._chunk = int(ceil(len(self._img) / self._number_of_workers))

    def apply(self):
        self._create_workers()
        self._start_workers()
        self._join_workers()

    def _create_workers(self):
        processed_rows = 0
        worker = self._create_worker(0, processed_rows, self._chunk, None, None, None, None)
        self._workers.append(worker)
        processed_rows += self._chunk
        for n in range(1, self._number_of_workers-1):
            worker = self._create_worker(
                n, processed_rows, self._chunk,
                worker.bottom_border, worker.bottom_border_lock,
                worker.last_row, worker.last_row_lock
            )
            self._workers.append(worker)
            processed_rows += self._chunk
        worker = self._create_worker(
            self._number_of_workers-1, processed_rows, len(self._img) - processed_rows,
            worker.bottom_border, worker.bottom_border_lock,
            worker.last_row, worker.last_row_lock
        )
        worker.bottom_border = None
        worker.bottom_border_lock = None
        self._workers.append(worker)

    def _create_worker(
            self, n: int, start: int, chunk: int,
            first_row: _WorkerRow, first_row_lock: Lock,
            top_border: _WorkerRow, top_border_lock: Lock
    ) -> _ConvWorker:
        print(f'worker {n} {start}-{start + self._chunk}')

        # noinspection PyUnresolvedReferences
        result: _WorkerResult = self._manager.result(chunk, len(self._img[0]))
        last_row = self._manager.row(len(self._img[0])+2)
        last_row_lock = self._lock_manager.Lock()
        bottom_border = self._manager.row(len(self._img[0])+2)
        bottom_border_lock = self._lock_manager.Lock()

        return _ConvWorker(
            n, self._iterations, self._img[start:start + chunk], self._matrix, result,
            first_row, last_row, first_row_lock, last_row_lock,
            top_border, bottom_border, top_border_lock, bottom_border_lock
        )

    def _start_workers(self):
        for worker in self._workers:
            worker.start()

    def _join_workers(self):
        for worker in self._workers:
            worker.join()
            if self._new_img is None:
                self._new_img = np.concatenate((worker.result.get(),))
            else:
                self._new_img = np.concatenate((self._new_img, worker.result.get()))

    def save_result(self):
        Image.fromarray(self._new_img).save(Conv.RESULT_FILE_NAME)
