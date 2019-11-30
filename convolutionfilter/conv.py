import numpy as np
from PIL import Image


class Conv:

    def __init__(self, img: np.array, matrix: np.array) -> None:
        self._img = img
        self._matrix = matrix
        self._new_img = []

    def save_filtered_image(self):
        self._apply()
        Image.fromarray(np.asarray(self._new_img)).save("result.jpg")

    def _apply(self):
        y_range = range(1, len(self._img) - 1)
        x_range = range(1, len(self._img[0]) - 1)
        for i in y_range:
            new_row = []
            for j in x_range:
                new_pixel = self._process_pixel(i, j)
                new_row.append(new_pixel)
            self._new_img.append(new_row)

    def _process_pixel(self, y: int, x: int):
        total = np.array([0, 0, 0], dtype=np.int16)
        weights = 0
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                weights += self._matrix[i+1][j+1]
                total += self._matrix[i+1][j+1] * self._img[y+i][x+i]

        return (total / weights).astype('uint8')
