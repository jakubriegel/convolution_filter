import sys
import time
from typing import Callable

from convolutionfilter.api import conv_from_file, MATRIX


def current_time() -> int:
    return int(time.time() * 1000)


def measure_time(start: int) -> int:
    return current_time() - start


def run_timed(task: Callable[[], None]):
    start = current_time()
    task()
    elapsed = measure_time(start)
    print(elapsed)


def app(img_file: str, matrix: str, workers: int, iterations: int, timed: str):
    def run(): conv_from_file(img_file, MATRIX[matrix], workers, iterations)

    if timed is not None:
        run_timed(run)
    else:
        run()


if __name__ == '__main__':
    app(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] if len(sys.argv) == 6 else None)
