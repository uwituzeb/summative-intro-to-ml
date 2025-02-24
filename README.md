# Pathway Finder

## Overview

This project aims to develop a machine learning model to predict career recommendations for high school students according to their interests, grades, extracurriculars, personality traits and favorite subjects.

## Problem statement

In Rwanda, high school students face significant challenges due to an inadequate career guidance system. This issue has led to limited career exploration, with students not being able to follow their passions because they are often not aware of what career options are available. A lot of students ask themselves, ‘what now?’ when they finish high school as they are rarely sure how to proceed. For more information, refer to this [project proposal](https://docs.google.com/document/d/1AUzYTCkMhIEOueQVTzMemrS-elA2W0bPxFy0d7dMXP4/edit?usp=sharing)

## Dataset

To create the classification models, [dataset](https://drive.google.com/file/d/1XC-gWLDLPuGyY9sepqH1tSXfd3Jm7aiJ/view?usp=sharing) was used. The dataset was synthetically created as there were no publicly available datasets that align with the project especially within the context of Rwanda. The dataset contains 9 columns of data with variables or parameters that help in predicting student career paths. The target variable is the recommended career, which is a multiclass classification.


## Findings

| Training Instance | Optimizer | Regularizer Used | Epochs | Early Stopping | Number of Layers | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|-------------------|-----------|------------------|--------|----------------|------------------|---------------|----------|----------|--------|-----------|
| Instance 1        | Default   | None             | 100    | False          | 4                | 0.001         | 0.56     | 0.48     | 0.49   | 0.49      |
| Instance 2        | Adam      | L2               | 100    | True           | 4                | 0.001         | 0.64     | 0.48     | 0.52   | 0.49      |
| Instance 3        | Adam      | L1               | 100    | True           | 4                | 0.001         | 0.63     | 0.46     | 0.50   | 0.48      |
| Instance 4        | SGD       | L2               | 100    | True           | 4                | 0.001         | 0.63     | 0.50     | 0.51   | 0.56      |
| Instance 5        | RMSProp   | L1_L2            | 100    | True           | 4                | 0.001         | 0.65     | 0.50     | 0.52   | 0.51      |

## Summary

### Neural network comparison

The models were trained for multi-class classification, with five neural network model instances and a logistic regression model. Out of all the model instances for neural networks, the best performing model was instance 5 which was using L1_L2 regularization and RMSProp optimizer, however it was still giving a low accuracy of 0.65. The worst performing model was instance 1 which used default hyperparameters, as expected. Instance 1 ended up giving an accuracy of 0.56.

If we are to compare according to f1 score and precision, instance 4(SGD and L2) was the best with the instance 5 being a close second as the f1 scores were almost tied. And for, recall the best performing model is instance 2(Adam and L2) and instance 5 which had a recall of 0.52. This continues to prove that instance 5 was the best combination.

It is clear that adding regularization helps improve models as the unregularized model performed worst. 

Overall, the difference was not very high. All the models shared a 4-layer network architecture, trained with a learning rate of 0.001 for up to 100 epochs, early stopping was also used except on the first instance to prevent overfitting.

### Justifications

- Instance 5 (RMSProp) performed best because RMSProp is known for handling adaptive learning rates well, making it effective for non-stationary objectives like career recommendation.
- Adam (Instances 2 & 3) performed well but slightly lower because while Adam combines momentum and adaptive learning rates, it sometimes over-adapts and doesn't generalize as well as RMSProp in certain cases.
- SGD (Instance 4) had good precision but lower recall because it updates weights more conservatively, which helps with generalization but may slow convergence.
- Models with Early Stopping (Instances 2, 3, 4, and 5) outperformed Instance 1 because early stopping prevents overfitting, ensuring better generalization. Instance 1 (no early stopping) had the worst accuracy (0.56) because training for too long led to overfitting, making the model less reliable on test data.
- Instance 5 (L1_L2) worked best by preventing overfitting while keeping important features, this is because comining L1 and L2 helps in generalization. L2 (Instances 2 & 4) improved recall by keeping weights balanced and preventing the model from focusing too much on some features, while L1 (Instance 3) lowered precision by removing some useful career-related features, because it forces some weights to 0.





### Logistic regression comparison

The logistic regression model performed best overall, with an accuracy of 0.93, f1-score of 0.91 and precision of 0.92. This shows that the logistic model consistently outperformed the neural network models. The main reason for this is that the logistic regrssion model was evaluated on a larger test size of 600 compared to neural networks which were tested on only 200 samples. Also, the neural networks may have overcomplicated the problem leading to overfitting.

The logistic regression model was trained with the following hyperparameters:

- multi_class='multinomial' to enable the model to handle multi-class classification
- solver='lbfgs' for optimization
- max_iter=1000 which sets a maximum of 1000 iterations
- random_state=42 for reproducibility of results

## Video presentation

[Video presentation](https://www.loom.com/share/1b4bcfd5c5e84a939d812c64a3c7c985?sid=aaf58404-1b20-4253-acb9-0d9e2b028c77)

## Instructions for running the notebook

### Prerequisites:

Ensure you have Python 3.x installed with the following libraries:

  - scikit-learn (for LogisticRegression and metrics)
  - numpy (for data handling)
  - joblib (for model loading)

2. Run the notebook:

   Open the notebook in Jupyter Notebook and execute all cells to train models, evaluate performance and save the best model to saved_models directory
   
4. Load and use the best model:
   The best performing model was logistic regression and to run it we pass the test data and       the path of the model
   
    ```
    model_path = 'saved_models/logistic_regression_model.pkl'
    make_predictions(model_path, x_test_scaled)
    ```
   

### Load saved model

The best performing model was logistic regression and to run it we pass the test data and the path of the model
```
model_path = 'saved_models/logistic_regression_model.pkl'
make_predictions(model_path, x_test_scaled)
```

