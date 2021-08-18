# Solar_Weather_Forecast

The final goal of this project is to predict the flare class and the location with a certainty
in the future. We are using images of the Sun of different wavelength for our purpose. Right
now, we are able to classify the solar flare in the images available with a F1 score of 0.95.
We can use Grad-Cam to get location of the flare. 

## Download Data and Preprocess

Download AIA files of 94A, 171A and 304 A from "http://jsoc.stanford.edu/data/aia/synoptic"
with one-hour cadence from 2010-05-13 to 2016-01-01.

```python
python download.py
```

Process data to downsample(average) it from 1024x1024 to 256x256, compiled as (3, 256, 256)

```python
python process.py
```

Parse the label available, used in process.py

```python
python parse_label.py
```

## Load Data

- One method is to load everything into memory by using `Dataset`, the number of images will be
limited by memory but good for initial testing.
- Another method is to load through `ImageFolder`, this will load from disk and can be quick after 
the first iteration. Use for real training. Notice this method require a specific directory 
structure (where folder name is class name). Please check out online documentation.

## Model

Used a variation of densenet as our model. Treat it as a multi-class problem and use CrossEntrophy 
and Adam Optimizer.

## Train

To train the model, using google colab for testing. Use NOTS available for Rice to utilize GPU. 
Otherwise, use AWS or other clouding computing platform for GPU acceleration.

```python
python train.py
```

The trained model is at `saved_model.pth`.

## Grad-CAM

Visualize the heatmap of the last of the convolution layer. It is the activation of the images when
passed through the model.

## Future

- [YOLO](https://github.com/ultralytics/yolov5), You Only Look Once. It will be very useful for recongnizing solar flares. However, it
will need labels in format of class, x, y, width, height (in fractions). You can see documentation
for more information. In order to make the labeling faster, you can go to https://www.makesense.ai/.

- The standard of image tools available must have 3 or 1 channels which is very annoying. I have made 
it possible to use files with any channels. With that, we can concantenate the different channels of 
different timestamps for the future prediction problem.