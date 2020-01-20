# convolution filter
## about
Concurrent implementation of images convolution filter. 
The application uses Python `multiprocessing` module for concurrency and `numpy.array` for image processing.

## table of contents
  * [about](#about)
  * [run it](#run-it)
  * [cli](#cli)
    + [filter mode](#filter-mode)
      - [info](#info)
      - [command](#command)
      - [arguments](#arguments)
      - [example](#example)
    + [benchmark mode](#benchmark-mode)
      - [info](#info-1)
      - [command](#command-1)
      - [arguments](#arguments-1)
      - [example](#example-1)
  * [algorithm](#algorithm)
  * [sample benchmark](#sample-benchmark)
    + [machine](#machine)
    + [parameters](#parameters)
    + [results](#results)
  * [credits](#credits)

## run it
To run the app first create the virtual environment:
```shell script
python -m venv cf-venv
```

Then start the environment:
```shell script
source cf-venv/bin/activate 
```

Next install dependencies:
```shell script
pip install -r requirements.txt
```

Start the app:
```shell script
python conv.py [ARGS]
```

## cli
### filter mode
#### info
In this mode the app applies convolution filer to given image.

#### command
```shell script
python conv.py [IMAGE] [BLUR] [NUMBER_OF_WORKERS] [MEASURE_TIME?]
```

#### arguments
* IMAGE - name of image file to apply filter on
* BLUR - choose blur matrix to use
* NUMBER_OF_WORKERS - number of workers (processes) to use
* MEASURE_TIME - optional, `t` for printing execution time

#### example
```shell script
python conv.py image.ppm blur1 8 1 t
```

### benchmark mode
#### info
In this mode the app applies convolution filer many times on consecutive numbers of workers.
Time results are being printed as `csv` to standard output. 

#### command
```shell script
python conv.py bench [IMAGE] [MAX_WORKERS] [MAX_ITERATIONS]
```

#### arguments
* IMAGE - name of image file to apply filter on
* MAX_WORKERS - maximal number of workers to run benchmark on
* MAX_ITERATIONS - maximal number of iteration to run benchmark on

#### example
```shell script
python conv.py bench image.ppm 12 50
```

## algorithm
tba

## sample benchmark
### machine
* Intel Core i7 8th gen 2.2Ghz (max. 4.1Ghz) 6x
* 32GB DDR4
* macOS Catalina

### parameters
```shell script
MAX_WORKERS=24
MAX_ITERATIONS=10
```
### results
![image](https://user-images.githubusercontent.com/32958017/72752923-cd76d500-3bc3-11ea-9373-f0a2a66bcaee.png)

## credits
The project was made by Jakub Riegel during Software Engineering course on Poznan University of Technology

![put logo](https://www.put.poznan.pl/themes/newputpoznan/images/logo.png)
