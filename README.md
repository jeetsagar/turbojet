# Prognostics Health Management on Turbofan Engine Data

CS230 Deep Learning project by Aditya Gulati and Jeetsagar Ghorai.

## Description of src/TF-gui

Integrated PHM ui with training, test and data pipeline. The ui can be invoked
using `python PHM.py`. The UI can be used to train models and test. This is the
recommended method for training and inference.

## Description of src/UKF-datagen

Generates the physics enhanced data using the Unscented Kalman Filter algorithm.
To create the UKF enhanced data, run `python UKFgen.py` followed by
`python UKFdatamake.py`.
