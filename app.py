import time
from typing import Callable

from convolutionfilter.api import conv_from_file, MATRIX


def current_time() -> int:
    return int(time.time() * 1000)


def measure_time(start: int) -> int:
    return current_time() - start


def run_timed(task: Callable[[], None]) -> int:
    start = current_time()
    task()
    elapsed = measure_time(start)
    return elapsed


def app() -> None:
    elapsed = run_timed(lambda: conv_from_file('img.jpg', MATRIX['blur1']))
    print(f'computation time {elapsed}ms')


if __name__ == '__main__':
    app()
