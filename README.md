
# Kaggle Tabular Playground Series - August

This project was developed for the Kaggle Tabular Playground Series (August). The goal is to predict failure probabilities given various numerical and categorical features using machine learning models. The evaluation metric used in this competition is **AUC/ROC**, which measures the area under the ROC curve (true positive rate vs. false positive rate).

## Project Overview

The competition provides a dataset containing a large number of features, some of which are categorical, and the objective is to predict the failure probability. The methodology includes a blend of Logistic Regression models, pre-processing techniques, and feature engineering.

### Key Highlights:
- **Predictive Model**: Blending multiple Logistic Regression models.
- **Pre-processing**: Missing value imputation, feature standardization, and categorical encoding.
- **Feature Engineering**: Extracting meaningful features and handling missing values.
- **Model Evaluation**: Using Stratified K Fold Cross Validation for model training and evaluation.
- **Post-processing**: Ensemble techniques were applied to improve prediction performance.

## Methodology

### Data Pre-processing
- **Missing Values Handling**:
  - Converted missing values in `measurement_3`, `measurement_4`, and `measurement_5` to binary features, allowing the model to capture their potential correlation with failure.
  - Used `log1p` transformation for `loading` feature to ensure a normal distribution for better prediction results.
  - One-hot encoded categorical features `attribute_0` and `attribute_1`.
  - Used regression and KNN imputation for missing values in `measurement_17`.

- **Standardization**: Standardized numerical features using `StandardScaler`.

### Model Architecture

Three Logistic Regression models were used in a blending approach:
- **Model 1**:
  - Features: `loading`, `measurement_17`, `measurement_2`, and missing value features (`m3_missing`, `m5_missing`).
- **Model 2**:
  - Features: `measurement_1`, `measurement_2`, `loading`, `measurement_17`.
- **Model 3**:
  - Features: `loading`, `measurement_17`, `measurement_2`, `m3_missing`, `m5_missing`.

### Hyperparameter Tuning
- Tuned the `C` and `Solver` parameters in Logistic Regression using grid search to achieve an average AUC score of approximately 0.592.

### Ensemble Model
- Combined the predictions of the three Logistic Regression models using a weighted ensemble approach (`0.4*rank0 + 0.3*rank1 + 0.3*rank2`) to improve final prediction performance.

## Results

- **Final Private Score**: Achieved a score of **0.5915** on the private leaderboard using the ensemble of Logistic Regression models.
- The model performance was slightly improved through careful tuning and post-processing techniques such as ranking the probabilities from individual models and blending them effectively.

## Repository Structure

```
├── data/
│   ├── train.csv        # Training dataset
│   ├── test.csv         # Testing dataset
├── src/
│   ├── preprocessing.py # Data pre-processing scripts
│   ├── train.py         # Model training scripts
│   ├── ensemble.py      # Post-processing and model blending
├── notebooks/           # Jupyter notebooks used for exploratory data analysis
├── README.md            # Project documentation
└── LICENSE              # License file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Yoppman/Kaggle-Tabular-Playground-Series-Aug.git
cd Kaggle-Tabular-Playground-Series-Aug
```

2. Install the required dependencies using Poetry:
```bash
poetry install
```

## Running the Project

1. **Data Preprocessing**: Prepare the dataset by running the pre-processing script.
```bash
python src/preprocessing.py
```

2. **Training the Models**: Train the logistic regression models.
```bash
python src/train.py
```

3. **Ensemble the Models**: Combine the predictions using the ensemble method.
```bash
python src/ensemble.py
```

## Conclusion

This project demonstrates the power of feature engineering and model blending techniques in predictive modeling. Through rigorous pre-processing, feature selection, and ensemble modeling, we were able to achieve a high score on the Kaggle leaderboard.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
