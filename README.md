 README #

Machine Learning Powered 'Question-type Classification'
==================

Tensorflow implementation of CNN models for classification

**Setup**

* Tensorflow, version = r0.12.1 (https://www.tensorflow.org/versions/r0.12/get_started/)
* Python 2.7

**Usage**:
```bash
Training:
Reads Training parameters from file config.ini
-Command Line Interface:
python train.py
 
Testing/serving:
-Command Line Interface:
   python predict.py
   
PS: The model might not be trained well due to:
* imbalanced data for classes
* lack of data and hence vocab

Can give better performance:
* if trained with sequential language models using RNNs(LSTMs)
* increase the training data(balanced)
* use pretrained vectors for word embeddings
* tune the training parameters well

```
