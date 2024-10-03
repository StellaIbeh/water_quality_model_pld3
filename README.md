# Water Potability Classification Project

## Overview
This project aimed to develop and compare multiple machine learning models for predicting water potability. The primary goal was to build convolutional neural networks (CNNs) to classify water samples as potable or non-potable based on a set of features. The features included pH, hardness, solids, chloramines, sulfate, and others. The dataset used for this project had a significant class imbalance, with more non-potable than potable samples, which influenced model performance.

## Project Contributors and Tasks
The project was completed with contributions from four members of the group, each responsible for different parts of the pipeline:

- ### Isaiah - Data Handling and Preprocessing:

Isaiah loaded the dataset and performed crucial preprocessing tasks:
Handled missing values in the 'ph', 'sulfate', and 'trihalomethanes' columns by filling in their mean.
Scaled the data using standard scaling to normalize the features.
Reshaped the data to prepare it for the CNN model.
Split the dataset into training (80%) and test sets (20%).


- ### Pierrette - Vanilla Model Implementation:

Pierrette implemented a simple sequential CNN model without using any optimizers or regularizers.
The model obtained the following results:
Training Accuracy: 72.52%
Validation Accuracy: 64.33%
Test Accuracy: 64.31%


- ### Adaobi Stella - L1 Regularized Model (Adam Optimizer):

Adaobi developed an L1 regularized model using the Adam optimizer.
Regularization was used to add a penalty proportional to the absolute value of the model weights to prevent overfitting.
The L1 regularized model obtained the following results:
Training Accuracy: 60.23%
Validation Accuracy: 62.80%
Test Accuracy: 62.80%


- ### Rene - L2 Regularized Model (RMSProp Optimizer):

Rene implemented an L2 regularized model using the RMSProp optimizer.
The L2 regularization helps penalize large weights and improves model generalization.
The L2 regularized model obtained the following results:
Training Accuracy: 60.79%
Validation Accuracy: 62.80%
Test Accuracy: 62.80%

## Summary of Model Performance and Findings
Three different models were developed, each with unique attributes:

1. Vanilla Model (No Regularization or Optimizer Customization)
The vanilla model was a basic sequential CNN model with no optimizers or regularizers applied.
Test Accuracy: 64.31%
Confusion Matrix:
lua

- [[288 124]
 [110 134]]

- Classification Report:
Precision for Non-Potable: 0.72
Recall for Non-Potable: 0.70
F1-score for Non-Potable: 0.71
Precision for Potable: 0.52
Recall for Potable: 0.55
F1-score for Potable: 0.53
Overall Accuracy: 64.33%
* Implication: The vanilla model performed better in predicting non-potable water but struggled to accurately classify potable water. This was likely due to the class imbalance in the dataset, where the model was more exposed to non-potable examples during training.

2. L1 Regularized Model (Adam Optimizer)
The L1 regularized model used L1 penalties to encourage sparsity in weights, potentially making the model more interpretable and less prone to overfitting.
Test Accuracy: 62.80%
Confusion Matrix:

- [[412   0]
 [244   0]]

- Classification Report:
Precision for Non-Potable: 0.63
Recall for Non-Potable: 1.00
F1-score for Non-Potable: 0.77
Precision for Potable: 0.00
Recall for Potable: 0.00
F1-score for Potable: 0.00
Overall Accuracy: 62.80%

* Implication: The L1 model showed that it was unable to correctly classify any potable water samples. This resulted in a perfect recall for non-potable water, but zero recall for potable water. This behavior indicates that the model was over-penalizing its weights, leading to underfitting, which was exacerbated by the class imbalance. The model learned to always predict the majority class (non-potable) to minimize loss.

3. L2 Regularized Model (RMSProp Optimizer)
The L2 regularized model used L2 penalties, which help to reduce large weight values, leading to better generalization.
Test Accuracy: 62.80%

- Confusion Matrix:

- [[412   0]
 [244   0]]

- Classification Report:
Precision for Non-Potable: 0.63
Recall for Non-Potable: 1.00
F1-score for Non-Potable: 0.77
Precision for Potable: 0.00
Recall for Potable: 0.00
F1-score for Potable: 0.00
Overall Accuracy: 62.80%

* Implication: Similar to the L1 model, the L2 model also failed to classify any potable water samples. The confusion matrix and classification report were identical to the L1 model, indicating that it learned to always predict the majority class (non-potable). The class imbalance and strong regularization likely prevented the model from learning features specific to the minority class.

## Analysis and Conclusion
Impact of Class Imbalance
The dataset used in this project suffered from a significant class imbalance, with a larger proportion of non-potable water samples compared to potable samples. This imbalance had a major impact on model training and evaluation:

* Vanilla Model: The vanilla model without any regularization or optimizers performed marginally better in predicting both classes, though it still struggled with correctly classifying potable water. The confusion matrix showed that while it could classify some potable samples, it still misclassified a large number of them.

L1 and L2 Regularized Models: Both the L1 and L2 models failed to predict any potable water samples correctly. The models learned to predict only the majority class (non-potable), which was the result of the class imbalance combined with strong regularization. This indicates that the regularization penalty forced the models to converge towards a simpler solution that only favored the majority class.