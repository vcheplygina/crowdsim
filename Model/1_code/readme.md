The 1_code folder contains the baseline and multi-task models used based on three different CNN architectures. Here is a description of the files:

- 0_baseline.py: The baseline model predicts a binary label (malignant or not) from a skin lesion image. The model is built on a convolutional base and extended further by adding specific layers. As an encoder, we used the VGG16, Inception v3, and ResNet50 convolutional base.


The remaining code files contain helper functions used by the above files.
