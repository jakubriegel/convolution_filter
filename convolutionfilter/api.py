from typing import List

import numpy as np
from PIL import Image

from convolutionfilter.conv import Conv


def conv_from_file(img_file: str, matrix: List[List[int]]) -> None:
    img = np.asarray(Image.open(img_file))
    conv(img, matrix, 8)


def conv(img: np.ndarray, matrix: List[List[int]], number_of_workers: int = 1) -> None:
    f = Conv(img, matrix, number_of_workers)
    f.apply()
    f.save_result()


MATRIX = {
    "blur1": [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ],
    "blur2": [
        [1, 1, 1],
        [1, 2, 1],
        [1, 1, 1]
    ],
    "blur3": [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ],
    "sharpen1": [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ],
    "sharpen2": [
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ],
    "sharpen3": [
        [1, -2, 1],
        [-2, 5, -2],
        [1, -2, 1]
    ]
}
