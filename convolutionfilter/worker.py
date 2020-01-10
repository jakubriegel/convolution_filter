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
            self, n: int, runs: int, data: np.ndarray, matrix: np.ndarray, result: _WorkerResult
    ) -> None:
        super().__init__(name=f'worker {n}')
        self._runs = runs
        self._data = data
        self._matrix = matrix
        self._result = result

        height = len(data)
        width = len(data[0])
        self._height = range(1, height+2)
        self._width = range(1, width+1)
        self._current_iter = np.pad(self._data, ((1, 1), (1, 1), (0, 0)), 'edge')
        self._next_iter = np.pad(self._data, ((1, 1), (1, 1), (0, 0)), 'edge')
        self._matrix_sum = self._matrix.sum()

    def run(self) -> None:
        self._process_iterations()

    def _process_iterations(self):
        for _ in range(0, self._runs):
            self._process_next_iteration()
        self._result.copy_from(self._current_iter)

    def _process_next_iteration(self):
        self._process_rows()
        self._switch_iterations()

    def _switch_iterations(self):
        t = self._current_iter
        self._current_iter = self._next_iter
        self._next_iter = t

    def _process_rows(self) -> None:
        self._process_row(0, self._current_iter[0:1])
        # self._process_row(self._height+1, self._current_iter[0:1])
        for i in self._height:
            self._process_row(i, self._current_iter[i - 1:i + 2])

    def _process_row(self, i: int, row: np.ndarray) -> None:
        for j in self._width:
            if len(row) == 3:
                self._next_iter[i][j] = self._process_pixel(row[0:3, j - 1:j + 2])
            else:
                self._next_iter[i][j] = self._process_pixel(row[0:2, j - 1:j + 2])

    def _process_pixel(self, pixel: np.ndarray) -> np.ndarray:
        return np.array([self._calculate_pixel(pixel[0:3, 0:3, i]) for i in (0, 1, 2)], dtype=np.uint8)

    def _calculate_pixel(self, value: np.ndarray) -> int:
        if len(value) == 3:
            return (value * self._matrix).sum() / self._matrix_sum
        else:
            return (value * self._matrix[:2]).sum() / self._matrix[:2].sum()
