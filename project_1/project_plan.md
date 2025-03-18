# Project I - Image classification with convolutional neural networks
**Team**: Szymon Smagowski, Jerzy Kraszewski

## Dataset

[Dataset - CINIC 10](https://www.kaggle.com/datasets/mengcius/cinic10)

The dataset contains ~270k images of 10 classes. Those are 32x32 color (RGB) images. The dataset comes with established division into training and test sets of same sizes (90k each). 

## General approach

We will start with a simple CNN model and the code for training, validation and testing. The code will be modular so that the models can be easily swapped. Then we will test different established architectures starting with a simple model (e.g. VGG) and then move to more complex ones (e.g. ResNet, EfficientNet). On top of that we will check the influence of hyperparameters, possibliy the same for all models. on the obtained results (both training and regularization). We will also investigate the influence of data augmentation techniques on the obtained results. Finally, we will apply ensemble methods (hard/soft voting, stacking) to the models.


## Plan

- [ ] Load the dataset
- [ ] Preprocess the dataset
- [ ] Implement simple CNN
- [ ] Train the model (wihout hyperparameter tuning)
- [ ] Evaluate the model
- [ ] Add data augmentation techniques
- [ ] Add hyperparameter tuning
- [ ] Add regularization techniques
- [ ] Evaluate the model
- [ ] Modularize the code for easy model and hyperparameters swapping
- [ ] Add VGG
- [ ] Add ResNet and EfficientNet
- [ ] Prepare the report
- [ ] Prepare the preparation

## Report outline:

1. Problem introduction (image classification)
2. Dataset description
3. Very short review of literature (available architectures)
4. Justified choice of architectures
5. Methodology description:
    - Data preprocessing
    - Training process
    - Hyperparameter tuning
    - Evaluation
6. Short code description (used libraries, etc., how to use it)
7. Results
8. Conclusions

## Notes:
- at least 2 hyper-parameters related to training process
- at least 2 hyper-parameters related to regularization
- at least X data augmentation techniques from the following groups:
  - standard operations (where x=3)
  - more advanced data augmentation techniques like mixup, cutmix, cutout (where x=1)
- constant seed for reproducibility
- implement one method dedicated to few-shot learning
- consider application of ensemble (hard/soft voting, stacking)
- preferred pre-trained models
- compare to achieved accuracy [CIFAR-10 accuracy for methods](https://benchmarks.ai/cifar-10) and 
- adhere to statistical significance
- variance not less important than mean

