
# Technical Specification: Credit Decision Simulator Jupyter Notebook

## 1. Notebook Overview

This Jupyter Notebook provides a detailed, interactive simulation of a consumer credit decision model. It demonstrates an end-to-end credit scoring process using a logistic regression model, synthetic applicant data, and explainability features. The notebook is designed for risk teams, model validators, compliance officers, and underwriters to understand the mechanics of compliant and explainable credit decisions.

### Learning Goals

Upon completing this notebook, users will be able to:
*   Understand the core mathematical formulation of a logistic regression model for credit default prediction.
*   Identify and apply necessary feature engineering steps for credit risk variables.
*   Generate and analyze synthetic applicant data with realistic distributions.
*   Train a logistic regression model and interpret its coefficients in an economic context.
*   Implement a credit decision logic based on predicted default probabilities and predefined thresholds.
*   Generate specific adverse action reasons using SHAP-style explainability for denied applications, adhering to ECOA requirements.
*   Evaluate model performance using key metrics such as AUC-ROC, calibration plots, and confusion matrices, aligning with SR 11-7 validation guidance.
*   Appreciate the importance of input validation and model stability in credit decision systems.

## 2. Code Requirements

### List of Expected Libraries

*   `numpy` (for numerical operations)
*   `pandas` (for data manipulation)
*   `sklearn.model_selection` (for data splitting)
*   `sklearn.preprocessing.StandardScaler` (for feature scaling)
*   `sklearn.linear_model.LogisticRegression` (for the credit scoring model)
*   `sklearn.metrics` (for AUC-ROC, confusion matrix, and calibration)
*   `matplotlib.pyplot` (for plotting)
*   `seaborn` (for enhanced visualizations)
*   `shap` (for model explainability)

### List of Algorithms or Functions to be Implemented

1.  **`generate_synthetic_data(num_samples: int, random_state: int) -> pd.DataFrame`**: Generates a DataFrame of synthetic credit applicant data including `income`, `credit_score`, `dti`, `tenure`, and a binary `default` target. The target variable generation should simulate a relationship that a logistic model can capture, with a realistic default rate (e.g., around 10-20%).
2.  **`apply_feature_transformations(data: pd.DataFrame) -> pd.DataFrame`**: Applies the specified feature transformations:
    *   `log_income` = $\ln(\max(\text{income}, 1))$
    *   `credit_score_scaled` = $(\text{credit\_score} - 680) / 100$
    *   `dti_clipped` = $\min(\text{dti}, 0.65)$
    *   `tenure` (used as-is)
3.  **`train_logistic_regression_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression`**: Trains a `sklearn.linear_model.LogisticRegression` model with `penalty='l2'`, `C=1/0.01` (for $\lambda=0.01$), `solver='lbfgs'`, `max_iter=1000`, and `class_weight='balanced'`.
4.  **`initialize_shap_explainer(model: LogisticRegression, X_transformed_train: pd.DataFrame) -> shap.LinearExplainer`**: Initializes a `shap.LinearExplainer` for the trained logistic regression model.
5.  **`validate_inputs(income: float, credit_score: int, dti: float, tenure: int) -> None`**: Validates input parameters according to the business rules (PRD f_004). Raises `ValueError` for invalid inputs.
6.  **`predict_default_probability(model: LogisticRegression, income: float, credit_score: int, dti: float, tenure: int) -> float`**: Predicts the default probability for a single applicant after applying feature transformations and input validation.
7.  **`make_credit_decision(probability: float) -> str`**: Applies the decision logic based on the probability of default:
    *   `p < 0.10`: "APPROVED"
    *   `0.10 <= p < 0.20`: "REVIEW_REQUIRED"
    *   `p >= 0.20`: "DENIED"
8.  **`generate_adverse_action_reasons(explainer: shap.LinearExplainer, model: LogisticRegression, income: float, credit_score: int, dti: float, tenure: int) -> list[str]`**: Generates a list of plain-English adverse action reasons for a denied application. This function should:
    *   Use the SHAP `explainer` to get feature contributions for the given input.
    *   Identify the top 3 features contributing most negatively (pushing towards denial).
    *   Formulate specific reasons based on the feature values and predefined thresholds (e.g., "Credit score X below recommended minimum (640)", "Debt-to-income ratio Y% exceeds guideline (43%)", "Employment history Z months below minimum (12)").
9.  **`plot_roc_curve(y_true: pd.Series, y_pred_proba: np.ndarray) -> None`**: Generates and displays an AUC-ROC curve.
10. **`plot_calibration_curve(y_true: pd.Series, y_pred_proba: np.ndarray) -> None`**: Generates and displays a calibration plot (reliability diagram).
11. **`plot_confusion_matrix(y_true: pd.Series, y_pred_binary: np.ndarray) -> None`**: Generates and displays a confusion matrix using a specified threshold (e.g., $p \ge 0.15$ for predicting default).

### Visualization like charts, tables, plots that should be generated

1.  **Synthetic Data Description Table**: `df.describe()` output.
2.  **Model Coefficients Table**: A table displaying the trained model's coefficients and intercept.
3.  **SHAP Force Plot (conceptual)**: A bar chart visualization showing feature contributions (SHAP values) for a single prediction, specifically for a denied case.
4.  **AUC-ROC Curve**: A line plot showing the Receiver Operating Characteristic curve with the AUC score.
5.  **Calibration Plot (Reliability Diagram)**: A scatter plot or line plot showing predicted probabilities versus observed default rates.
6.  **Confusion Matrix Heatmap**: A heatmap visualizing True Positives, True Negatives, False Positives, and False Negatives.

## 3. Notebook sections (in detail)

---

### Section 1: Introduction to Credit Decision Simulation

This section introduces the purpose of the notebook: to simulate an interactive credit decision process using a logistic regression model. It highlights the importance of explainability and compliance in lending.

---

### Section 2: Core Logic - The Logistic Regression Model

The heart of our credit decision simulator is a logistic regression model, which estimates the probability of default. Logistic regression is chosen for its interpretability and probabilistic output, crucial for regulatory compliance.

The probability of default $P(y=1 | x)$ is given by the sigmoid function applied to a linear combination of features:

$$P(y=1 | x) = \sigma(\beta_0 + \beta_1x_1 + \beta_2x_2 + \beta_3x_3 + \beta_4x_4)$$

Where $\sigma(z)$ is the logistic (sigmoid) function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

The features ($x_i$) are transformations of raw applicant data:
*   $x_1$: Transformed annual income
*   $x_2$: Transformed credit score
*   $x_3$: Transformed debt-to-income ratio (DTI)
*   $x_4$: Employment tenure in months

**Note on Coefficients:** For this simulation, we will aim to train a model that conceptually aligns with the expected coefficient signs and magnitudes from the technical specification, which are crucial for economic soundness.

---

### Section 3: Feature Engineering

Before training the model, raw input features are transformed to improve model performance and interpretability.

*   **Income**: A logarithmic transformation is applied to annual income to account for diminishing returns and normalize its skewed distribution. We take $\ln(\max(\text{income}, 1))$ to handle zero or negative income safely.
    $$x_1 = \ln(\max(\text{income}, 1))$$
*   **Credit Score**: The FICO credit score is standardized to a mean of 0 and a standard deviation of 1. Based on the technical specification, a historical mean of $680$ and standard deviation of $100$ are used for scaling.
    $$x_2 = \frac{\text{credit\_score} - 680}{100}$$
*   **Debt-to-Income Ratio (DTI)**: The DTI is clipped at $0.65$ (65%) to prevent extreme outliers from disproportionately influencing the model and to reflect business thresholds.
    $$x_3 = \min(\text{dti}, 0.65)$$
*   **Employment Tenure**: This feature is used directly as months of employment, as a linear relationship is expected.
    $$x_4 = \text{tenure}$$

---

### Section 3.1: Feature Transformation Function
```
Function: apply_feature_transformations
Description: Takes a pandas DataFrame with raw input features (income, credit_score, dti, tenure) and returns a new DataFrame with the transformed features (log_income, credit_score_scaled, dti_clipped, tenure_raw).
```

### Section 3.2: Example of Feature Transformation
```
Code:
Create a sample DataFrame with one applicant's raw data.
Call `apply_feature_transformations` on this sample DataFrame.
Display the resulting transformed features.
```

### Section 3.3: Explanation of Feature Transformation
This execution demonstrates how the raw applicant data is prepared for the logistic regression model by applying the specified transformations. Observe how `income` is logged, `credit_score` is scaled, and `dti` is clipped to their respective processed forms.

---

### Section 4: Synthetic Data Generation

To simulate a credit portfolio, we generate a synthetic dataset. This allows us to work with realistic data distributions without involving any Personally Identifiable Information (PII) and to demonstrate model behavior. The dataset will include `income`, `credit_score`, `dti`, `tenure`, and a binary `default` target variable.

*   **Income**: Log-normally distributed, reflecting common income patterns.
*   **Credit Score**: Normally distributed, centered around typical FICO scores.
*   **DTI**: Beta-distributed or similar, to capture a range of debt burdens.
*   **Tenure**: Exponentially distributed or similar, to show varying employment histories.
*   **Default**: Generated based on a probabilistic function of the features, to simulate a realistic relationship a logistic model would learn, aiming for an overall default rate of approximately 10-20%.

We will generate 5000 samples for our synthetic dataset. A `random_state` will be set for reproducibility.

---

### Section 4.1: Synthetic Data Generation Function
```
Function: generate_synthetic_data
Description: Generates a pandas DataFrame of synthetic credit applicant data.
    - `num_samples`: 5000
    - `random_state`: 42 (for reproducibility)
    - `income`: np.random.lognormal(mean=11.0, sigma=0.8, size=num_samples) (approx $60k-$100k)
    - `credit_score`: np.random.normal(loc=680, scale=80, size=num_samples), clipped to [300, 850]
    - `dti`: np.random.beta(a=2, b=7, size=num_samples) * 1.5, clipped to [0, 1.5]
    - `tenure`: np.random.randint(low=0, high=240, size=num_samples) (0-20 years)
    - `default`: Generated probabilistically based on a linear combination of the raw features (or their transformed versions) plus some noise, passed through a sigmoid function, aiming for a default rate around 10-20%. For example, `default_prob = 1 / (1 + np.exp(- (0.00005*income - 0.005*credit_score + 2*dti - 0.01*tenure + 5.0)))`, then `default = (np.random.rand(num_samples) < default_prob).astype(int)`.
```

### Section 4.2: Generate and Inspect Synthetic Data
```
Code:
Call `generate_synthetic_data(num_samples=5000, random_state=42)`.
Display the first 5 rows of the generated DataFrame (`.head()`).
Display descriptive statistics of the DataFrame (`.describe()`).
Calculate and display the overall default rate.
```

### Section 4.3: Explanation of Synthetic Data
The output shows the structure and summary statistics of our synthetic credit applicant data. We can see the ranges and distributions of `income`, `credit_score`, `dti`, and `tenure`. The overall default rate indicates the base risk level within this simulated portfolio. This data will be used to train and validate our credit scoring model.

---

### Section 5: Model Training

We will now train our logistic regression model using the synthetic data. The goal is to learn the coefficients ($\beta_i$) that best predict the probability of default based on the transformed features. The model will be trained using Maximum Likelihood Estimation with L2 regularization to prevent overfitting and ensure numerical stability.

---

### Section 5.1: Model Training Function
```
Function: train_logistic_regression_model
Description: Trains a Logistic Regression model.
    - Uses `sklearn.model_selection.train_test_split` to split the data (e.g., 80% train, 20% test, `random_state=42`, `stratify=y`).
    - Applies `apply_feature_transformations` to both training and test sets.
    - Initializes `sklearn.linear_model.LogisticRegression` with specified parameters: `penalty='l2'`, `C=1/0.01`, `solver='lbfgs'`, `max_iter=1000`, `class_weight='balanced'`.
    - Fits the model on the transformed training data.
    - Returns the trained model, transformed training features, and actual training targets.
```

### Section 5.2: Train the Model
```
Code:
Split the `synthetic_df` into `X` (features) and `y` (target `default`).
Call `train_logistic_regression_model(X, y)`.
Display the learned coefficients (`model.coef_`) and intercept (`model.intercept_`) in a readable format (e.g., pandas Series or DataFrame).
```

### Section 5.3: Explanation of Model Training
The model has been successfully trained. The displayed coefficients indicate the learned relationship between each transformed feature and the log-odds of default. For example, a negative coefficient for `log_income` implies that higher income (or $\ln(\text{income})$) is associated with a lower log-odds of default, which aligns with economic intuition (Tech Spec PROP-008). The intercept represents the baseline log-odds of default when all features are zero.

---

### Section 6: Credit Decision Logic

Once the default probability is predicted, a credit decision is made based on predefined business rules and thresholds. This ensures consistent and automated decision-making.

The decision logic is as follows:

$$
\text{decision}(p) =
\begin{cases}
\text{APPROVED}, & p < 0.10 \\
\text{REVIEW\_REQUIRED}, & 0.10 \le p < 0.20 \\
\text{DENIED}, & p \ge 0.20
\end{cases}
$$

Additionally, robust input validation is critical to ensure the model receives meaningful data. Inputs must adhere to specific ranges:
*   `income > 0`
*   `300 <= credit_score <= 850`
*   `0 <= dti <= 1.5`
*   `tenure >= 0`

---

### Section 6.1: Prediction and Decision Functions
```
Function: validate_inputs
Description: Checks if input parameters are within valid business ranges. Raises ValueError if invalid.

Function: predict_default_probability
Description: Takes raw applicant inputs, performs validation, applies feature transformations, and returns the predicted default probability using the trained model.

Function: make_credit_decision
Description: Takes a default probability and returns the categorical decision (APPROVED, REVIEW_REQUIRED, DENIED).
```

### Section 6.2: Simulate Credit Decisions for Example Applicants
```
Code:
Define three example applicants (A, B, C) with different profiles:
    Applicant A (likely APPROVED): income=80000, credit_score=750, dti=0.25, tenure=60
    Applicant B (likely REVIEW_REQUIRED): income=50000, credit_score=680, dti=0.40, tenure=24
    Applicant C (likely DENIED): income=30000, credit_score=580, dti=0.55, tenure=10

For each applicant:
    Call `validate_inputs`.
    Call `predict_default_probability`.
    Call `make_credit_decision`.
    Print the applicant's profile, predicted probability, and decision.
```

### Section 6.3: Explanation of Credit Decisions
This simulation demonstrates how different applicant profiles lead to varying default probabilities and corresponding credit decisions. Applicant A, with a strong profile, is approved due to low default risk. Applicant B, with a borderline profile, requires review. Applicant C, with higher risk factors, is denied. This highlights the consistent application of the decision rules based on the model's output.

---

### Section 7: Explainability - Adverse Action Reasons with SHAP

For denied applications, the Equal Credit Opportunity Act (ECOA) requires providing specific reasons. SHapley Additive exPlanations (SHAP) is a method to explain individual predictions by showing how much each feature contributed to the final outcome. We use SHAP to generate plain-English adverse action reasons.

A `shap.LinearExplainer` will be initialized with our trained logistic regression model and the transformed training data. This explainer will calculate SHAP values for individual predictions.

---

### Section 7.1: SHAP Explainer Initialization and Adverse Action Reason Function
```
Function: initialize_shap_explainer
Description: Initializes and returns a `shap.LinearExplainer` for the trained model.

Function: generate_adverse_action_reasons
Description: Takes an applicant's raw inputs, the SHAP explainer, and the trained model.
    - Validates inputs.
    - Transforms inputs.
    - Calculates SHAP values for the single applicant.
    - Identifies the top features contributing to a denial.
    - Generates 1-3 specific, plain-English reasons (e.g., "Credit score X below recommended minimum (640)", "Debt-to-income ratio Y% exceeds guideline (43%)", "Employment history Z months below minimum (12)") based on feature values and their contributions. If no specific reasons are strong enough, a generic "Overall credit profile indicates high default risk" reason should be included.
```

### Section 7.2: Generate Adverse Action Reasons for a Denied Applicant
```
Code:
Using Applicant C (denied) from the previous section:
    Initialize the SHAP explainer using the trained model and transformed training data.
    Call `generate_adverse_action_reasons` with Applicant C's data and the explainer.
    Print the generated reasons.
    Optionally, visualize the SHAP values for Applicant C as a bar chart (or conceptual force plot if direct SHAP plot isn't feasible in spec) to show feature contributions.
```

### Section 7.3: Explanation of Adverse Action Reasons
The output provides specific, actionable reasons why Applicant C's application was denied. These reasons are derived from the individual feature contributions to the model's prediction, ensuring transparency and compliance with ECOA requirements. The SHAP values would quantitatively show which features pushed the probability towards denial the most.

---

### Section 8: Model Performance - Discrimination (AUC-ROC)

Model validation is critical for regulatory compliance (SR 11-7). Discrimination refers to the model's ability to distinguish between defaulters and non-defaulters. The Area Under the Receiver Operating Characteristic curve (AUC-ROC) is a widely used metric for this. An AUC-ROC value of $0.5$ indicates random performance, while $1.0$ indicates perfect discrimination. The non-functional requirement (NFR) for this model is an AUC-ROC > $0.75$.

---

### Section 8.1: AUC-ROC Plotting Function
```
Function: plot_roc_curve
Description: Calculates and plots the AUC-ROC curve using `sklearn.metrics.roc_curve` and `sklearn.metrics.roc_auc_score`.
    - Takes true labels (`y_true`) and predicted probabilities (`y_pred_proba`).
    - Uses `matplotlib.pyplot` and `seaborn` for visualization.
    - Displays the calculated AUC score on the plot.
```

### Section 8.2: Evaluate and Plot AUC-ROC
```
Code:
Use the test set (`X_test_transformed`, `y_test`) from the model training phase.
Predict probabilities on the `X_test_transformed` using the trained model.
Call `plot_roc_curve(y_test, y_pred_proba_test)`.
```

### Section 8.3: Explanation of AUC-ROC
The AUC-ROC curve and score demonstrate the model's ability to correctly rank applicants by their default risk. A higher AUC score (ideally above 0.75 as per NFRs) indicates good discriminatory power, meaning the model is effective at distinguishing between applicants who will default and those who will not.

---

### Section 9: Model Performance - Calibration

Calibration assesses whether the predicted probabilities match the actual observed default rates. For example, if the model predicts a 10% probability of default for 100 applicants, then approximately 10 of those applicants should actually default. Good calibration is crucial for setting accurate risk-based pricing and for regulatory scrutiny (SR 11-7 outcomes analysis).

---

### Section 9.1: Calibration Plotting Function
```
Function: plot_calibration_curve
Description: Generates and displays a reliability diagram (calibration plot).
    - Takes true labels (`y_true`) and predicted probabilities (`y_pred_proba`).
    - Uses `sklearn.calibration.calibration_curve` to calculate `fraction_of_positives` and `mean_predicted_value`.
    - Plots these values, along with a perfect calibration line, using `matplotlib.pyplot`.
```

### Section 9.2: Evaluate and Plot Calibration
```
Code:
Use the test set (`y_test`, `y_pred_proba_test`).
Call `plot_calibration_curve(y_test, y_pred_proba_test)`.
```

### Section 9.3: Explanation of Calibration Plot
The calibration plot shows how well the model's predicted probabilities align with observed default rates. The closer the plotted line is to the diagonal (perfectly calibrated line), the more reliable the probabilities are. Deviations indicate under- or over-prediction of risk in certain probability ranges. This plot is essential for ensuring the model's risk estimates are trustworthy.

---

### Section 10: Model Performance - Confusion Matrix

The confusion matrix provides a detailed breakdown of model predictions versus actual outcomes, especially useful for understanding decision-making at a specific threshold. For credit decisions, we are particularly interested in True Positives (correctly identified defaulters) and False Negatives (missed defaulters), as these have significant financial implications. We will use a threshold of $P(\text{default}) \ge 0.15$ to classify an applicant as a "predicted defaulter" for this analysis.

---

### Section 10.1: Confusion Matrix Plotting Function
```
Function: plot_confusion_matrix
Description: Generates and displays a confusion matrix heatmap.
    - Takes true labels (`y_true`) and binary predicted labels (`y_pred_binary`).
    - Uses `sklearn.metrics.confusion_matrix` to compute the matrix.
    - Uses `seaborn.heatmap` for visualization, with appropriate labels (True Negatives, False Positives, False Negatives, True Positives).
```

### Section 10.2: Evaluate and Plot Confusion Matrix
```
Code:
Use the test set (`y_test`, `y_pred_proba_test`).
Define a classification threshold, e.g., `threshold = 0.15`.
Convert `y_pred_proba_test` to binary predictions (`y_pred_binary_test`) using this threshold.
Call `plot_confusion_matrix(y_test, y_pred_binary_test)`.
```

### Section 10.3: Explanation of Confusion Matrix
The confusion matrix provides a granular view of the model's classification performance. We can see how many applicants were correctly approved (True Negatives - non-defaulters predicted as non-defaulters) or correctly denied (True Positives - defaulters predicted as defaulters). Crucially, it highlights False Positives (applicants denied who would not have defaulted) and False Negatives (applicants approved who subsequently defaulted), allowing for a balanced assessment of the model's impact on business risk and customer experience.

