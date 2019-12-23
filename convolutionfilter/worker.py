from multiprocessing import Process
from multiprocessing.managers import BaseManager

import numpy as np


class _WorkerResult:

    def __init__(self, width: int, height: int) -> None:
        self._data = np.empty((width, height, 3), dtype=np.uint8)

    def _set_row(self, y: int, value: np.ndarray):
        self._data[y] = value[1:-1]

    def copy_from(self, source: np.ndarray) -> None:
        for i in range(1, len(source)-1):
            self._set_row(i-1, source[i])

    def get(self): return self._data


class _WorkersManager(BaseManager):

    def __init__(self) -> None:
        super().__init__()
        self.start()


_WorkersManager.register('result', _WorkerResult)


class _ConvWorker(Process):

    def __init__(
            self, n: int, data: np.ndarray, matrix: np.ndarray, result: _WorkerResult
    ) -> None:
        super().__init__(name=f'worker {n}')
        self._data = data
        self._matrix = matrix
        self._result = result

        height = len(data)
        width = len(data[0])
        self._height = range(1, height)
        self._width = range(1, width)
        self._current_gen = np.pad(self._data, ((1, 1), (1, 1), (0, 0)), 'edge')
        self._next_gen = np.empty((height+2, width+2, 3), dtype=np.uint8)

    def run(self) -> None:
        self._process_rows()
        self._result.copy_from(self._next_gen)

    def _process_rows(self) -> None:
        for i in self._height:
            self._process_row(i, self._current_gen[i-1:i+2])

    def _process_row(self, i: int, row: np.ndarray) -> None:
        for j in self._width:
            self._next_gen[i][j] = self._process_pixel(row[0:3, j-1:j+2])

    def _process_pixel(self, pixel: np.ndarray) -> np.ndarray:
        return np.array([self._calculate_pixel(pixel[0:3, 0:3, i]) for i in (0, 1, 2)], dtype=np.uint8)

    def _calculate_pixel(self, value: np.ndarray) -> int:
        return (value * self._matrix).sum() / self._matrix.sum()
