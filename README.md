# Head Tracking
This code creates a virtual webcam that can be used in conference calls. One of the features implemented is face tracking. The webcam will follow the head of the speaker centering on the image.

## External Software

Download and Install OBS Studio:

https://obsproject.com

## Python

This code is tested to work with Python 3.9. At the moment it doesn't work on Mac Silicon. In order to avoid the problem use the terminal in Rosetta mode.

## Requirements

```
pip install -r requirements.txt
```

## Run and optional arguments

```
python app_trackFace.py
```

```
python app_trackFace.py --help
  -h, --help            show this help message and exit
  -a ACTIVE_AREA, --area ACTIVE_AREA
                        Active area for tracking
  -s AVERAGE_SMOOTH, --smooth AVERAGE_SMOOTH
                        Smooth tracking taking average position last N points.
                        Default is 60
  -ow PREFERRED_WIDTH, --output_width PREFERRED_WIDTH
                        Threshold for prediction. A number between 0 and 1.
                        default is 0.5
  -oh PREFERRED_HEIGHT, --output_height PREFERRED_HEIGHT
                        Threshold for prediction. A number between 0 and 1.
                        default is 0.5
```
