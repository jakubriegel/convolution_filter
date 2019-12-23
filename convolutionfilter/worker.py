from multiprocessing import Process
from multiprocessing.managers import BaseManager
from typing import List

import numpy as np


class _WorkerResult:

    def __init__(self, width: int, height: int) -> None:
        self._data = np.empty((width, height, 3), dtype=np.uint8)

    def set_row(self, y: int, value: np.ndarray):
        self._data[y] = value

    def copy_from(self, source: np.ndarray) -> None:
        for i in range(0, len(source)):
            self.set_row(i, source[i])

    def get(self): return self._data


class _WorkersManager(BaseManager):

    def __init__(self) -> None:
        super().__init__()
        self.start()


_WorkersManager.register('result', _WorkerResult)


class _ConvWorker(Process):

    def __init__(
            self, n: int, data: np.ndarray, matrix: List[List[int]], result: _WorkerResult
    ) -> None:
        super().__init__(name=f'worker {n}')
        self._data = data
        self._matrix = matrix
        self._result = result

        self._height = len(data)
        self.width = len(data[0])
        self._current_gen = self._data
        self._next_gen = self._empty_gen()

    def _empty_gen(self): return np.empty((self._height, self.width, 3), dtype=np.uint8)

    def run(self) -> None:
        self._process_rows()
        self._result.copy_from(self._next_gen)

    def _process_rows(self) -> None:
        new_row = self._process_row(self._current_gen[0:1])
        self._next_gen[0] = new_row
        for i in range(1, len(self._current_gen)):
            new_row = self._process_row(self._current_gen[i-1:i+2])
            self._next_gen[i] = new_row

    def _process_row(self, row: np.ndarray) -> np.ndarray:
        new_row = np.empty((self.width, 3), dtype=np.uint8)
        new_row[0] = self._process_pixel(row[0:4, 0:1])
        for i in range(1, self.width):
            new_row[i] = self._process_pixel(row[0:4, i-1:i+2])
        return new_row

    def _process_pixel(self, pixel: np.ndarray) -> np.ndarray:
        total = np.array([0, 0, 0], dtype=np.int16)
        weights = 0
        for i in range(0, len(pixel)):
            for j in range(0, len(pixel[0])):
                weight = self._matrix[i][j]
                weights += weight
                total += weight * pixel[i][j]

        return (total / weights).astype('uint8')
