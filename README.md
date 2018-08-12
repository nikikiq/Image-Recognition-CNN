# Image Recognition of Handwritten Integers (0-9)
### Data Pre-process
1. Classes are well-balanced in training set.
2. Most digits located in the middle. Different levels of blur, rotation, shifting and scaling issues
exist. To deal with them and improve model's tolerance of noises, random rotation, zooming and
shifting is applied on training data.

### Model Selection
1. Application Package: Tensorflow, Keras
2. Model: Convolutional neural network(CNN)
3. Model Architecture: See project report for details

### Model Enhancement:
1. Using cross entropy instead of classification accuracy since it reflects how good the weights
are chosen, not just how many correct predictions are made.
2. Increasing the epoch numbers from 30-100
3. Adding more convolutional layers to implement a simplified version of VGG-16.
4. Skipping the normalisation after each pooling
5. Dropout before final layer to avoid over-fitting
