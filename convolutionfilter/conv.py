from multiprocessing import Process, Manager
from typing import List, Tuple

import numpy as np
from PIL import Image


class Conv:
    EXTENSION = '.jpg'
    RESULT_FILE_NAME = f'result{EXTENSION}'

    def __init__(self, img: np.ndarray, matrix: List[List[int]], number_of_workers: int) -> None:
        self._img = img
        self._matrix = matrix
        self._number_of_workers = number_of_workers

        self._new_img = []
        self._manager = Manager()
        self._workers: List[Tuple[_ConvWorker, list]] = []
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
        result = self._manager.list()
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
            self._new_img = self._new_img + list(worker[1])

    def save_result(self):
        img_array = np.asarray(self._new_img)
        Image.fromarray(img_array).save(Conv.RESULT_FILE_NAME)


class _ConvWorker(Process):

    def __init__(self, n: int, data: np.ndarray, matrix: List[List[int]], result: List[np.ndarray]) -> None:
        super().__init__(name=f'worker {n}')
        self._data = data
        self._matrix = matrix
        self._result = result

    def run(self) -> None:
        self._process_rows()

    def _process_rows(self):
        y_range = range(1, len(self._data) - 1)
        x_range = range(0, len(self._data[0]) - 1)
        for i in y_range:
            new_row = []
            for j in x_range:
                new_pixel = self._process_pixel(i, j)
                new_row.append(new_pixel)
            self._result.append(np.asarray(new_row))

    def _process_pixel(self, y: int, x: int) -> np.uint8:
        total = np.array([0, 0, 0], dtype=np.int16)
        weights = 0
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                weights += self._matrix[i+1][j+1]
                total += self._matrix[i+1][j+1] * self._data[y+i][x+i]

        return (total / weights).astype('uint8')
