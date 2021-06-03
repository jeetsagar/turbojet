# PyTorch Implementation

## Train

Train a model using the command `python torchtrain.py`. If the optional argument
`--restore` is provided, the file specified after `--load-file` will be loaded
and the training will be resumed. Use the flag `--cuda` to train using GPU
acceleration if CUDA is available. If the argument `--units` is omitted, all dev
units available in the data file are used for training.

```
optional arguments:
  -h, --help            show this help message and exit
  --traindata TRAINDATA
                        path to training data
  --units [UNITS ...]   units used for training
  --save-dir SAVE_DIR   model directory
  --overwrite           overwrite old models
  --load-file LOAD_FILE
                        model location
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size
  -e NUM_EPOCHS, --num-epochs NUM_EPOCHS
                        number of epochs
  --sequence            use sequence model
  --restore             restore pre-trained model
  --cuda                use cuda
```

## Test

Test a model using the command `python torchtest.py`. The checkpoint specified
by the argument `--load-file` is loaded. The units specified by `--units` are
used for testing. If the `--units` argument is omitted, all test units available
in the data file are used for testing.

```
optional arguments:
  -h, --help            show this help message and exit
  --testdata TESTDATA   path to test data
  --load-file LOAD_FILE
                        model location
  --units [UNITS ...]   units to use
  --sequence            use sequence model
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size
  --cuda                use cuda
```

The output of the model will be written to the file specified by the argument
`--output`. This file can be loaded as follows.

```python
import numpy as np

data = np.load('./output.npz')
predicted = data['output']
```
