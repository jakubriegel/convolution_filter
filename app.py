from convolutionfilter.api import conv_from_file, MATRIX


def app() -> None:
    conv_from_file('img.jpg', MATRIX['blur1'])


if __name__ == '__main__':
    app()
