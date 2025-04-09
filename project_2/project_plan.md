# Project II - Speech Commands Classification with Transformers

**Team:** Szymon Smagowski, Jerzy Kraszewski

## Dataset

### Speech Commands Dataset
[speech-dataset](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data)

The dataset contains audio recordings of spoken words, organized in categories representing different commands (such as "yes", "no", "up", "down", etc.) plus special classes for "silence" and "unknown" commands. Each recording is a short (around 1 second) audio file sampled at 16kHz.

## General approach
We will start by exploring the dataset and implementing appropriate audio preprocessing and feature extraction techniques. We will then implement and compare different network architectures, with a focus on Transformer-based models. We will investigate how different parameters affect the performance of these models. Special attention will be given to handling the "silence" and "unknown" classes, which may require unique approaches. We will evaluate our models using confusion matrices and appropriate performance metrics.

## Plan

- Load and explore the dataset
- Implement audio preprocessing (e.g., spectrograms, MFCCs)
- Create baseline non-Transformer model (e.g., CNN, RNN)
- Implement a Transformer architecture
- Train initial models on a subset of classes (e.g., "yes" and "no")
- Evaluate and compare model performances
- Expand to include more command classes
- Develop special handling for "silence" and "unknown" classes
- Investigate parameter influence on model performance
- Generate and analyze confusion matrices
- Prepare the report
- Prepare the presentation

## Report outline:

### Problem introduction (speech command classification)
### Dataset description
### Methodology description:

- Audio representation and feature extraction
- Model architectures
- Training process
- Parameter selection and optimization
- Special handling for "silence" and "unknown" classes

### Implementation details (used libraries, etc., how to use it)
### Results

- Quantitative evaluation
- Confusion matrix analysis
- Parameter influence analysis

### Conclusions and future work

## Notes:

- Test different network architectures (at least one must be a Transformer)
- Investigate influence of parameters on results
- Present and discuss confusion matrix
- For efficiency, start with subset of classes ("yes" and "no")
- Special attention to "silence" and "unknown" classes
- Test different approaches for handling special classes:
  - Separate network for their recognition
  - Under/oversampling techniques
  - Other specialized methods

- Ensure reproducibility with constant seeds
- Consider statistical significance of results
- Compare Transformer performance with other approaches
- Document any efficiency/accuracy tradeoffs made for computational constraints
