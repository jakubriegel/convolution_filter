from multiprocessing import Process
from multiprocessing.managers import BaseManager
from typing import List

import numpy as np


class _WorkerResult:

    def __init__(self, width: int, height: int) -> None:
        self._data = np.empty((width, height, 3), dtype=np.uint8)

    def set(self, y: int, x: int, value: np.ndarray):
        self._data[y][x] = value

    def set_row(self, y: int, value: np.ndarray):
        self._data[y] = value

    def get(self): return self._data


class _WorkersManager(BaseManager):

    def __init__(self) -> None:
        super().__init__()
        self.start()


_WorkersManager.register('result', _WorkerResult)


class _ConvWorker(Process):

    def __init__(
            self,
            n: int, data: np.ndarray,
            matrix: List[List[int]], result: _WorkerResult
    ) -> None:
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
            new_row = np.empty((len(self._data[0]), 3), dtype=np.uint8)
            for j in x_range:
                new_row[j] = self._process_pixel(i, j)
            self._result.set_row(i, new_row)

    def _process_pixel(self, y: int, x: int) -> np.ndarray:
        total = np.array([0, 0, 0], dtype=np.int16)
        weights = 0
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                weights += self._matrix[i+1][j+1]
                total += self._matrix[i+1][j+1] * self._data[y+i][x+i]

        return (total / weights).astype('uint8')
