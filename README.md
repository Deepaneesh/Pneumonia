# Pneumonia

Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing symptoms like:

-   Cough (which may produce phlegm or mucus)

-   Fever

-   Chills

-   Difficulty breathing

-   Chest pain, especially when breathing or coughing

-   Fatigue

### Causes:

Pneumonia can be caused by different types of microorganisms, including:

-   Bacteria (e.g., Streptococcus pneumoniae)

-   Viruses (e.g., influenza, COVID-19)

-   Fungi (especially in people with weakened immune systems)

### Types:

-   Community-acquired pneumonia (CAP) – most common, contracted outside of healthcare settings.

-   Hospital-acquired pneumonia (HAP) – develops during a hospital stay.

-   Ventilator-associated pneumonia (VAP) – occurs in people using ventilators.

-   Aspiration pneumonia – happens when food, drink, vomit, or saliva is inhaled into the lungs.

### Risk Factors:

-   Age (infants and elderly)

-   Weakened immune system

-   Chronic illnesses (like asthma, diabetes, heart disease)

-   Smoking

-   Hospitalization, especially with ventilator use

### Treatment:

-   Bacterial pneumonia is usually treated with antibiotics.

-   Viral pneumonia may be treated with antiviral medications or supportive care.

-   Rest, fluids, and oxygen therapy may also be recommended.

Most healthy people recover from pneumonia, but it can be serious or even life-threatening, especially in high-risk individuals.

# Pneumonia Detection using Chest X-Ray Images

**Problem Statement:**\
Pneumonia is a serious respiratory infection that requires timely diagnosis for effective treatment. Manual analysis of chest X-ray images is time-consuming and prone to human error. The goal is to develop a deep learning model, specifically using **ResNet50**, to automate the classification of chest X-ray images into **pneumonia** and **normal** cases.

**Approach:**

-   **Data Preprocessing:** Grayscale X-ray images resized to 224×224 pixels and converted to 3-channel RGB to match ResNet50’s input requirements.

-   **Normalization:** Pixel values were scaled between 0 and 1.

-   **Modeling:** A pre-trained ResNet50 model was fine-tuned for binary classification.

-   **Class Imbalance:** Addressed through class weighting and data augmentation.

-   **Evaluation:** Used confusion matrices and key metrics including Accuracy, F1-Score, and Cohen’s Kappa.

# Dataset

1.  Collected from **Kaggle**.

2.  Structured into three image folders: train, validation, and test.

3.  **Labels are provided in CSV files** mapping image filenames to binary classes:

    `0` → Normal

    `1` → Pneumonia

# Model

## ResNet50

**ResNet50** is a deep Convolutional Neural Network **(CNN)** with **50 layers**, introduced by Microsoft Research in 2015 in the paper "Deep Residual Learning for Image Recognition." It belongs to the family of Residual Networks (ResNets), which introduced the concept of skip connections (or residual connections).

ResNet uses identity shortcut connections that skip one or more layers. These connections help prevent the vanishing gradient problem and make it easier to train very deep networks.

## Architecture Overview

-   Total: 50 layers

-   Consists of:

    1.  1 convolution layer (Conv1)

    2.  16 **residual blocks** (using bottleneck design)

    3.  Each residual block = 3 layers (Conv → BN → ReLU)

## Common Hyperparameters (and their purpose)

| Hyperparameter | Description |
|------------------------------------|------------------------------------|
| input_shape | Size of input image. For ResNet50, usually (224, 224, 3) |
| weights | Set to "imagenet" to use pre-trained weights. |
| include_top | If False, removes the default final Dense layers, allowing customization. |
| pooling | When include_top=False, set to 'avg' to use global average pooling. |
| learning_rate | Controls the step size for gradient descent. Common values: 1e-3 to 1e-5. |
| optimizer | Algorithm to minimize the loss. Often Adam, SGD, or RMSprop. |
| batch_size | Number of samples processed before the model updates weights. |
| epochs | Number of full passes through the training data. |
| dropout_rate | Regularization technique to prevent overfitting. Typical values: 0.3 – 0.5. |
| data_augmentation | Adds variations to training images (rotation, zoom, etc.) to improve generalization. |
| class_weights | Used to handle class imbalance by assigning higher importance to minority class. |

## Model Summary

-   **ResNet50** is powerful for image classification and avoids overfitting better than traditional CNNs.

-   You customize only the final layers when doing transfer learning.

-   Tune hyperparameters like learning rate, dropout, and batch size to improve performance.

-   Combine with **augmentation** and **class weights** for better generalization in imbalanced datasets like yours.

#  Class Imbalance Detection & Mitigation

##  Detection of Class Imbalance

To detect class imbalance in the training set, the distribution of labels (Normal vs. Pneumonia) was analyzed using a frequency count and visualized with bar plots. The analysis revealed a significant imbalance—pneumonia cases were far more frequent than normal cases. This imbalance can bias the model towards predicting the majority class, reducing its ability to correctly detect the minority class.

## ️ Mitigation Strategies Used

To address this issue, the following techniques were employed:

1. **Data Augmentation for Minority Class**:
   - Using `ImageDataGenerator`, we applied transformations such as rotation, zooming, shifting, and flipping to artificially increase the number of samples in the minority class.
   - This helps expose the model to a wider variety of samples, improving generalization.

2. **Class Weighting**:
   - The `class_weight` parameter in the `model.fit()` function was used to assign a higher weight to the minority class.
   - This forces the model to penalize misclassification of minority class samples more during training.



## Evaluation Summary

To assess the performance of the ResNet-50 model for pneumonia detection using chest X-ray images, we evaluated it on the training, validation, and testing datasets. The metrics used include **Accuracy**, **Cohen’s Kappa**, and **F1 Score**—each offering a distinct perspective on the model's effectiveness, robustness, and ability to generalize across data splits.

## Model Evaluation Results

| Dataset    | Accuracy | Cohen's Kappa | F1 Score |
|------------|----------|---------------|----------|
| Training   | 0.9003   | 0.0046        | 0.8533   |
| Validation | 0.7424   | 0.0000        | 0.6326   |
| Testing    | 0.6250   | 0.0000        | 0.4808   |

### Interpretation & Insights

From the evaluation metrics above, we can draw several important observations:

-   **Training Set**:
    -   The model shows high accuracy (90.03%) and strong F1 Score (0.8533), which indicates that it fits the training data well.
    -   However, the Cohen’s Kappa is very low (0.0046), suggesting that the agreement between predicted and actual labels may not be much better than random guessing. This often occurs when there is class imbalance or overfitting.
-   **Validation Set**:
    -   The drop in accuracy (74.24%) and F1 Score (0.6326) compared to the training set reveals that the model’s performance deteriorates on unseen data.
    -   The Cohen’s Kappa is 0.0000, reinforcing the concern that the model may not be generalizing well, and might be overfitting to the training set.
-   **Test Set**:
    -   The performance further declines with an accuracy of 62.50% and a F1 Score of 0.4808.
    -   Again, a Cohen’s Kappa of 0.0000 shows no meaningful agreement between the model's predictions and true labels on test data.

### Key Takeaways

-   The model likely suffers from overfitting, where it learns the training data too well but fails to generalize to validation and test sets.
-   The extremely low Cohen’s Kappa values across all datasets indicate that class imbalance could be severely affecting the performance. The model may be biased toward the majority class.
-   The drop in F1 Score from training to testing suggests that the model struggles to balance precision and recall when evaluated on new, unseen images.

### future Improvements

To improve model performance and generalization:

-   Apply better class balancing techniques (e.g., SMOTE, class weighting, oversampling minority class).
-   Use data augmentation to introduce variability and reduce overfitting.
-   Implement regularization methods such as dropout and L2 regularization.
-   Explore fine-tuning deeper layers of ResNet-50 or try other architectures.
