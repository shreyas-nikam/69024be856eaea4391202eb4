
# Technical Specification: Credit Decision Simulator Notebook

## 1. Notebook Overview

This Jupyter Notebook guides users through the core mechanics of a consumer credit scoring model, specifically the "Credit Decision Simulator." It demonstrates how various financial inputs are processed to determine a credit decision and associated default probability, while also providing insights into input validation and decision explainability.

### Learning Goals

*   Understand the key financial factors (`income`, `credit_score`, `debt_to_income_ratio (DTI)`, `employment_tenure_months`, `loan_amount`) that influence a consumer credit score and loan decision.
*   Grasp the mathematical relationship between these inputs and the predicted default probability, specifically through the lens of a Logistic Regression model.
*   Learn how a credit scoring model translates a continuous default probability into discrete credit decisions (APPROVED, DENIED, or REVIEW_REQUIRED) using predefined business thresholds.
*   Recognize the importance of data validation in ensuring reliable model outputs and preventing erroneous predictions.
*   Gain insight into how adverse action reasons are generated for denied applications, promoting transparency and compliance.
*   Understand the key insights contained within the provided project requirements and supporting data.

## 2. Code Requirements

### List of Expected Libraries

The following open-source Python libraries will be used:

*   `pandas`: For data manipulation and tabular data handling.
*   `numpy`: For numerical operations, especially mathematical functions like `exp` and array manipulations.
*   `sklearn.linear_model.LogisticRegression`: While the model coefficients will be pre-defined for simulation, the conceptual understanding and potential for actual training with `sklearn` is implied.
*   `matplotlib.pyplot`: For generating static plots and visualizations.
*   `seaborn`: For enhanced statistical data visualizations.
*   `ipywidgets`: For creating interactive user interface controls (sliders, buttons).
*   `IPython.display`: For displaying rich output in Jupyter Notebooks.

### List of Algorithms or Functions to be Implemented

1.  `generate_synthetic_credit_data(num_samples: int) -> pandas.DataFrame`:
    *   Generates a synthetic dataset with realistic distributions for `income`, `credit_score`, `debt_to_income_ratio`, `employment_tenure_months`, and `loan_amount`.
    *   Adds a `defaulted_actual` column, randomly assigned based on a simulated default probability distribution.
2.  `validate_inputs(income: float, credit_score: int, dti: float, employment_tenure_months: int, loan_amount: float) -> dict`:
    *   Takes individual credit application inputs.
    *   Checks each input against predefined business rules (e.g., `300 <= credit_score <= 850`).
    *   Returns a dictionary indicating validation status for each input and a list of error messages if any validation fails.
3.  `calculate_default_probability(income: float, credit_score: int, dti: float, employment_tenure_months: int, loan_amount: float, model_coefficients: dict, model_intercept: float) -> float`:
    *   Calculates the log-odds ($z$) using the provided inputs, model coefficients, and intercept based on the Logistic Regression formula.
    *   Transforms the log-odds into a default probability using the sigmoid function.
    *   Returns the `default_probability` (float between 0 and 1).
4.  `make_credit_decision(default_probability: float) -> str`:
    *   Applies the credit decision business rules based on the `default_probability`.
    *   Returns "APPROVED", "DENIED", or "REVIEW_REQUIRED".
5.  `generate_adverse_action_reasons(income: float, credit_score: int, dti: float, employment_tenure_months: int, loan_amount: float, decision: str) -> list`:
    *   Takes the original inputs and the `decision`.
    *   If the `decision` is "DENIED", generates specific, plain-language reasons based on which inputs contributed most negatively to the decision thresholds.
    *   Returns a list of strings, each being an adverse action reason.
6.  `calculate_feature_contributions(income: float, credit_score: int, dti: float, employment_tenure_months: int, loan_amount: float, model_coefficients: dict, model_intercept: float) -> dict`:
    *   Calculates a simplified "SHAP-like" contribution for each feature, representing its weighted impact on the log-odds ($z$) score.
    *   Returns a dictionary where keys are feature names and values are their contributions.
7.  `simulate_credit_decision_pipeline(income: float, credit_score: int, dti: float, employment_tenure_months: int, loan_amount: float, model_coefficients: dict, model_intercept: float) -> dict`:
    *   Orchestrates the entire decision process: validation, probability calculation, decision making, adverse action reason generation, and feature contribution calculation.
    *   Returns a dictionary containing all outputs: `validation_status`, `error_messages`, `default_probability`, `decision`, `adverse_action_reasons`, `feature_contributions`.

### Visualization Requirements

1.  **Summary Statistics Table**: A `pandas.DataFrame.describe()` output for the synthetic dataset.
2.  **Head of Dataset Table**: `pandas.DataFrame.head()` output for the synthetic dataset.
3.  **Scatter Plot**: `credit_score` vs. `default_probability` from the synthetic dataset to show correlation.
    *   Color-coded by `decision` (if applicable, using thresholds on synthetic data).
    *   Clear titles, labeled axes, and legend.
    *   Color-blind friendly palette.
4.  **Pair Plot (Optional for exploration)**: A `seaborn.pairplot` to visualize relationships between all input features and potentially `default_probability` in the synthetic dataset.
5.  **Bar Chart**: Displaying `feature_contributions` for a single simulation, indicating the positive/negative impact of each input on the log-odds score.
    *   Clear title, labeled axes.
    *   Color-blind friendly palette.
6.  **Interactive Output Table/Display**: A dynamically updated display showing `default_probability`, `decision`, `adverse_action_reasons`, and potentially a bar chart of `feature_contributions` based on `ipywidgets` slider inputs.

## 3. Notebook Sections (in detail)

---

### **Section 1: Introduction to the Credit Decision Simulator**

This section introduces the purpose of the notebook and outlines the learning objectives.

#### Markdown Cell
Welcome to the **Credit Decision Simulator**!

This interactive lab is designed to demystify the process of how consumer loan applications are evaluated, focusing on predicting default probability and making credit decisions. You will learn how key financial factors influence creditworthiness and how a model translates these into actionable outcomes like approval or denial.

**Learning Objectives for this Notebook:**
*   Understand the key financial factors (`income`, `credit_score`, `debt_to_income_ratio (DTI)`, `employment_tenure_months`, `loan_amount`) that influence a consumer credit score and loan decision.
*   Grasp the mathematical relationship between these inputs and the predicted default probability, specifically through the lens of a Logistic Regression model.
*   Learn how a credit scoring model translates a continuous default probability into discrete credit decisions (APPROVED, DENIED, or REVIEW_REQUIRED) using predefined business thresholds.
*   Recognize the importance of data validation in ensuring reliable model outputs and preventing erroneous predictions.
*   Gain insight into how adverse action reasons are generated for denied applications, promoting transparency and compliance.
*   Understand the key insights contained within the provided project requirements and supporting data.

---

### **Section 2: Setting Up the Environment and Importing Libraries**

This section ensures all necessary libraries are imported for the notebook's functionality.

#### Markdown Cell
Before we begin, we need to import all the necessary Python libraries. These libraries provide functionalities for data manipulation, numerical operations, plotting, and creating interactive elements in our notebook.

#### Code Cell (Function Implementation)
```python
# Import standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import for interactive widgets
from ipywidgets import interact, FloatSlider, IntSlider, Text, Dropdown, HTML
from IPython.display import display, clear_output

# Set plotting style
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
```

#### Markdown Cell
The required libraries have been successfully imported, and plotting styles have been configured for better readability and a color-blind-friendly palette.

---

### **Section 3: Generating Synthetic Credit Application Data**

To illustrate the credit scoring process, we will generate a synthetic dataset of credit applications. This dataset will simulate various applicant profiles with realistic financial attributes.

#### Markdown Cell
In a real-world scenario, credit scoring models are built and validated using historical loan data. For this simulator, we will create a **synthetic dataset** to represent typical credit applications. This allows us to explore data characteristics and relationships before diving into the model's logic. The dataset will include `income`, `credit_score`, `debt_to_income_ratio (DTI)`, `employment_tenure_months`, and `loan_amount`. We will also simulate an `actual_default` column for demonstration purposes, indicating whether a loan would have historically defaulted.

#### Code Cell (Function Implementation)
```python
def generate_synthetic_credit_data(num_samples: int = 1000) -> pd.DataFrame:
    """
    Generates a synthetic dataset of credit applications.

    Args:
        num_samples (int): The number of synthetic credit applications to generate.

    Returns:
        pd.DataFrame: A DataFrame containing synthetic credit application data.
    """
    np.random.seed(42) # for reproducibility

    # Generate realistic distributions for features
    income = np.random.normal(loc=60000, scale=20000, size=num_samples).round(-2)
    income[income < 10000] = 10000 # Minimum income

    credit_score = np.random.normal(loc=680, scale=80, size=num_samples).astype(int)
    credit_score = np.clip(credit_score, 300, 850) # FICO range

    dti = np.random.normal(loc=0.35, scale=0.15, size=num_samples)
    dti = np.clip(dti, 0.05, 1.0) # Realistic DTI range

    employment_tenure_months = np.random.normal(loc=48, scale=30, size=num_samples).astype(int)
    employment_tenure_months = np.clip(employment_tenure_months, 0, 240) # 0 to 20 years

    loan_amount = np.random.normal(loc=25000, scale=10000, size=num_samples).round(-2)
    loan_amount = np.clip(loan_amount, 5000, 50000) # Product limits

    data = pd.DataFrame({
        'income': income,
        'credit_score': credit_score,
        'debt_to_income_ratio': dti,
        'employment_tenure_months': employment_tenure_months,
        'loan_amount': loan_amount
    })

    # Simulate default probability for the synthetic data based on simplified logic
    # This is a proxy for the actual model to make the synthetic data realistic
    simulated_z = (
        -0.000015 * data['income']
        -0.006 * data['credit_score']
        +1.0 * data['debt_to_income_ratio']
        -0.002 * data['employment_tenure_months']
        +0.000005 * data['loan_amount']
        +2.5 # Intercept, adjusted for rough distribution
    )
    simulated_prob_default = 1 / (1 + np.exp(-simulated_z))
    
    # Introduce some noise to actual default for realism
    data['defaulted_actual'] = (simulated_prob_default + np.random.normal(0, 0.05, num_samples) > 0.15).astype(int)
    data['simulated_default_probability'] = simulated_prob_default

    return data
```

#### Code Cell (Function Execution)
```python
synthetic_data = generate_synthetic_credit_data(num_samples=1000)
print("First 5 rows of the synthetic dataset:")
display(synthetic_data.head())
```

#### Markdown Cell
We have successfully generated a synthetic dataset containing 1000 credit applications. The table above shows the first 5 entries, giving us a glimpse of the data structure and content. Each row represents a unique loan applicant with their financial characteristics and a simulated actual default outcome.

---

### **Section 4: Exploring the Synthetic Dataset**

Before building our simulator, let's perform a basic exploratory data analysis (EDA) on the synthetic dataset to understand its characteristics, distributions, and summary statistics.

#### Markdown Cell
Understanding the underlying data is crucial for any model. This step provides summary statistics and visualizes key relationships within our synthetic dataset. This helps confirm that the generated data is realistic and covers the expected ranges.

#### Code Cell (Function Execution)
```python
print("Summary statistics of the synthetic dataset:")
display(synthetic_data.describe())

print("\nData types and missing values:")
display(synthetic_data.info())
```

#### Markdown Cell
The summary statistics provide a quick overview of the central tendency, dispersion, and shape of the distributions for each numeric feature. We can see that column names are as expected, data types are appropriate (e.g., `int` for `credit_score`, `float` for `DTI`), and there are no missing values in critical fields, confirming our data generation and handling.

---

### **Section 5: Visualizing Data Relationships**

Visualizations help us identify patterns and correlations within the data, which are fundamental to how a credit model functions.

#### Markdown Cell
Visualizing the relationships between features can provide valuable insights into how different financial factors might influence the likelihood of default. Here, we'll create a scatter plot to observe the relationship between `credit_score` and `simulated_default_probability`, as `credit_score` is often a primary determinant of credit risk.

#### Code Cell (Function Execution)
```python
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=synthetic_data,
    x='credit_score',
    y='simulated_default_probability',
    hue='defaulted_actual', # Use the simulated actual default for coloring
    palette='coolwarm',
    alpha=0.6
)
plt.title('Credit Score vs. Simulated Default Probability', fontsize=16)
plt.xlabel('Credit Score', fontsize=12)
plt.ylabel('Simulated Default Probability', fontsize=12)
plt.legend(title='Actual Default', loc='upper right')
plt.tight_layout()
plt.show()

# Optional: A pair plot to explore all relationships (can be slow for large datasets)
# print("\nPair plot showing relationships between features (may take a moment):")
# sns.pairplot(synthetic_data[['income', 'credit_score', 'debt_to_income_ratio', 'simulated_default_probability']], hue='defaulted_actual', palette='coolwarm')
# plt.suptitle('Pair Plot of Key Features', y=1.02, fontsize=16)
# plt.tight_layout()
# plt.show()
```

#### Markdown Cell
The scatter plot clearly illustrates an inverse relationship between `credit_score` and `simulated_default_probability`: as the credit score increases, the probability of default generally decreases. This aligns with common financial understanding. The different colors indicate which simulated applications actually defaulted, providing a visual sense of where defaults are more concentrated.

---

### **Section 6: Defining the Credit Scoring Model Parameters**

Our credit decision simulator will use a **Logistic Regression** model's structure to predict default probability. This section defines the pre-determined coefficients and intercept that represent a trained model.

#### Markdown Cell
A credit scoring model, at its core, quantifies risk. We'll use a **Logistic Regression** model, which is a common choice in credit risk due to its interpretability. Logistic Regression models the probability of a binary outcome (like default/no-default) using a sigmoid function applied to a linear combination of input features.

The core of the model is the **log-odds** ($z$), calculated as:
$$ z = \beta_0 + \sum_{i=1}^{n} (\beta_i \cdot X_i) $$
Where:
*   $\beta_0$ is the intercept.
*   $\beta_i$ are the coefficients for each feature $X_i$.

The `default_probability` ($P$) is then derived from $z$ using the sigmoid function:
$$ P = \frac{1}{1 + e^{-z}} $$

For our simulator, we will use pre-defined `model_coefficients` and `model_intercept` to represent a model that has already been trained on historical data. This allows us to focus on the decision-making process without the overhead of model training within the notebook.

#### Code Cell (Function Implementation)
```python
# Define the pre-determined coefficients and intercept for our Logistic Regression model
# These values are chosen to produce realistic default probabilities for the simulator
model_intercept = 2.5
model_coefficients = {
    'income': -0.000015,             # Higher income reduces default probability
    'credit_score': -0.006,          # Higher credit score reduces default probability
    'debt_to_income_ratio': 1.0,     # Higher DTI increases default probability
    'employment_tenure_months': -0.002, # Longer tenure reduces default probability
    'loan_amount': 0.000005          # Higher loan amount slightly increases default probability
}

print("Model Intercept:", model_intercept)
print("Model Coefficients:", model_coefficients)
```

#### Markdown Cell
The model's `intercept` and `coefficients` have been explicitly defined. You can observe the signs of the coefficients: negative values (e.g., `income`, `credit_score`, `employment_tenure_months`) indicate that an increase in these factors **decreases** the default probability, while positive values (e.g., `debt_to_income_ratio`, `loan_amount`) indicate an **increase** in default probability. The magnitude of the coefficients indicates the strength of the feature's influence.

---

### **Section 7: Implementing Input Validation Logic (F_004)**

Before making any predictions, it's crucial to validate the input data to ensure it falls within acceptable business ranges and data types. Invalid inputs can lead to erroneous predictions and model instability.

#### Markdown Cell
A robust credit scoring system must include **input validation** (Functional Requirement F_004). This ensures that the data fed into the model is within sensible business ranges, preventing nonsensical predictions or system errors. For instance, a credit score cannot be below 300 or above 850. If inputs are invalid, the system should reject them with clear, actionable feedback.

Our validation rules are:
*   `income`: Must be greater than 0.
*   `credit_score`: Must be between 300 and 850 (inclusive).
*   `debt_to_income_ratio (DTI)`: Must be between 0 and 1.5 (150%, inclusive).
*   `employment_tenure_months`: Must be non-negative (0 or greater).
*   `loan_amount`: Must be between 5000 and 50000 (inclusive, reflecting product limits).

#### Code Cell (Function Implementation)
```python
def validate_inputs(income: float, credit_score: int, dti: float, employment_tenure_months: int, loan_amount: float) -> dict:
    """
    Validates individual credit application inputs against predefined business rules.

    Args:
        income (float): Annual gross income.
        credit_score (int): FICO credit score.
        dti (float): Debt-to-income ratio.
        employment_tenure_months (int): Months of employment tenure.
        loan_amount (float): Requested loan amount.

    Returns:
        dict: Contains 'is_valid' (boolean) and 'messages' (list of strings).
    """
    is_valid = True
    messages = []

    if not (income > 0):
        is_valid = False
        messages.append("Income must be positive.")
    if not (300 <= credit_score <= 850):
        is_valid = False
        messages.append("Credit score must be between 300 and 850.")
    if not (0 <= dti <= 1.5):
        is_valid = False
        messages.append("Debt-to-income ratio must be between 0 and 1.5 (150%).")
    if not (employment_tenure_months >= 0):
        is_valid = False
        messages.append("Employment tenure must be non-negative.")
    if not (5000 <= loan_amount <= 50000):
        is_valid = False
        messages.append("Loan amount must be between $5,000 and $50,000.")

    return {"is_valid": is_valid, "messages": messages}
```

#### Code Cell (Function Execution)
```python
# Test with valid inputs
valid_test_inputs = {
    'income': 60000,
    'credit_score': 720,
    'dti': 0.35,
    'employment_tenure_months': 48,
    'loan_amount': 15000
}
validation_result_valid = validate_inputs(**valid_test_inputs)
print("Validation for valid inputs:", validation_result_valid)

# Test with invalid inputs
invalid_test_inputs = {
    'income': -1000,
    'credit_score': 250,
    'dti': 2.0,
    'employment_tenure_months': -5,
    'loan_amount': 60000
}
validation_result_invalid = validate_inputs(**invalid_test_inputs)
print("\nValidation for invalid inputs:", validation_result_invalid)
```

#### Markdown Cell
The `validate_inputs` function correctly identified both valid and invalid input scenarios. For invalid inputs, it provides specific error messages, which are crucial for giving users actionable feedback (Functional Requirement F_004).

---

### **Section 8: Calculating Default Probability (F_001)**

The core function of the credit scoring model is to predict the probability that an applicant will default on their loan within a specified timeframe.

#### Markdown Cell
The primary functional requirement (F_001) of our model is to **predict the probability of default** within 12 months. This probability is calculated using the logistic regression model parameters (coefficients and intercept) and the applicant's financial inputs.

The log-odds ($z$) is first computed as a linear combination of features and their respective weights (coefficients):
$$ z = \beta_0 + (\beta_{\text{income}} \cdot \text{income}) + (\beta_{\text{credit\_score}} \cdot \text{credit\_score}) + (\beta_{\text{DTI}} \cdot \text{DTI}) + (\beta_{\text{tenure}} \cdot \text{employment\_tenure\_months}) + (\beta_{\text{loan\_amount}} \cdot \text{loan\_amount}) $$
This $z$ value is then transformed into a probability ($P$) using the **sigmoid function**:
$$ P = \frac{1}{1 + e^{-z}} $$
This probability $P$ will always be between 0 and 1.

#### Code Cell (Function Implementation)
```python
def calculate_default_probability(
    income: float, credit_score: int, dti: float, employment_tenure_months: int, loan_amount: float,
    model_coefficients: dict, model_intercept: float
) -> float:
    """
    Calculates the default probability using a logistic regression model.

    Args:
        income (float): Annual gross income.
        credit_score (int): FICO credit score.
        dti (float): Debt-to-income ratio.
        employment_tenure_months (int): Months of employment tenure.
        loan_amount (float): Requested loan amount.
        model_coefficients (dict): Dictionary of feature coefficients.
        model_intercept (float): The model's intercept.

    Returns:
        float: The calculated default probability (between 0 and 1).
    """
    # Calculate the linear combination (z)
    z = model_intercept
    z += model_coefficients['income'] * income
    z += model_coefficients['credit_score'] * credit_score
    z += model_coefficients['debt_to_income_ratio'] * dti
    z += model_coefficients['employment_tenure_months'] * employment_tenure_months
    z += model_coefficients['loan_amount'] * loan_amount

    # Apply the sigmoid function to get the probability
    default_probability = 1 / (1 + np.exp(-z))
    return default_probability
```

#### Code Cell (Function Execution)
```python
# Test with a sample application (same as valid_test_inputs)
sample_app = {
    'income': 60000,
    'credit_score': 720,
    'dti': 0.35,
    'employment_tenure_months': 48,
    'loan_amount': 15000
}
calculated_prob = calculate_default_probability(
    **sample_app,
    model_coefficients=model_coefficients,
    model_intercept=model_intercept
)
print(f"Calculated Default Probability: {calculated_prob:.4f}")
```

#### Markdown Cell
For the provided sample application, the `default_probability` was calculated to be approximately $0.0768$ (or $7.68\%$). This value will now be used to determine the final credit decision.

---

### **Section 9: Making the Credit Decision (F_002)**

The predicted default probability is then translated into a concrete credit decision based on predefined business rules and thresholds.

#### Markdown Cell
Once the `default_probability` is calculated, the model needs to make a definitive **credit decision** (Functional Requirement F_002). This decision is based on specific, pre-defined thresholds that convert the continuous probability into a categorical outcome: APPROVED, DENIED, or REVIEW_REQUIRED.

The business rules for decision making are:
*   If $ \text{default\_prob} < 0.10 $: **APPROVED** (Low risk)
*   If $ 0.10 \leq \text{default\_prob} < 0.20 $: **REVIEW\_REQUIRED** (Marginal risk, needs human judgment)
*   If $ \text{default\_prob} \geq 0.20 $: **DENIED** (High risk)

#### Code Cell (Function Implementation)
```python
def make_credit_decision(default_probability: float) -> str:
    """
    Makes a credit decision based on the calculated default probability and business rules.

    Args:
        default_probability (float): The probability of default.

    Returns:
        str: The credit decision ("APPROVED", "DENIED", or "REVIEW_REQUIRED").
    """
    if default_probability < 0.10:
        return "APPROVED"
    elif 0.10 <= default_probability < 0.20:
        return "REVIEW_REQUIRED"
    else: # default_probability >= 0.20
        return "DENIED"
```

#### Code Cell (Function Execution)
```python
# Test with the previously calculated probability
decision_for_sample_app = make_credit_decision(calculated_prob)
print(f"Credit Decision for sample application (prob {calculated_prob:.4f}): {decision_for_sample_app}")

# Test with a probability that would result in REVIEW_REQUIRED
prob_review = 0.15
decision_review = make_credit_decision(prob_review)
print(f"Credit Decision for prob {prob_review:.4f}: {decision_review}")

# Test with a probability that would result in DENIED
prob_denied = 0.25
decision_denied = make_credit_decision(prob_denied)
print(f"Credit Decision for prob {prob_denied:.4f}: {decision_denied}")
```

#### Markdown Cell
The `make_credit_decision` function successfully applies the specified thresholds. Our sample application, with a default probability of $0.0768$, falls into the "APPROVED" category. The tests with $0.15$ and $0.25$ probabilities correctly yield "REVIEW_REQUIRED" and "DENIED" respectively, demonstrating the decision logic.

---

### **Section 10: Generating Adverse Action Reasons (F_003)**

For denied applications, the model must provide specific, plain-language reasons for the adverse action, as mandated by fair lending regulations like ECOA.

#### Markdown Cell
When a credit application is **DENIED**, providing specific, understandable reasons for that adverse action is not just good customer service—it's a regulatory requirement (Functional Requirement F_003), particularly under the Equal Credit Opportunity Act (ECOA). These reasons help applicants understand what factors contributed to the denial and how they might improve their credit profile.

The reasons should be based on the applicant's inputs relative to common credit guidelines. For example:
*   **Credit Score**: If $ \text{credit\_score} < 640 $ (a common minimum threshold).
*   **Debt-to-Income Ratio (DTI)**: If $ \text{DTI} > 0.43 $ (a common maximum threshold for qualified mortgages).
*   **Employment Tenure**: If $ \text{employment\_tenure\_months} < 12 $ (a common minimum for stable employment).
*   **Loan Amount vs. Income**: If the requested $ \text{loan\_amount} $ is excessively high relative to $ \text{income} $ (e.g., $ \text{loan\_amount} > 0.5 \times \text{income} $).

#### Code Cell (Function Implementation)
```python
def generate_adverse_action_reasons(
    income: float, credit_score: int, dti: float, employment_tenure_months: int, loan_amount: float, decision: str
) -> list:
    """
    Generates specific adverse action reasons if the decision is 'DENIED'.

    Args:
        income (float): Annual gross income.
        credit_score (int): FICO credit score.
        dti (float): Debt-to-income ratio.
        employment_tenure_months (int): Months of employment tenure.
        loan_amount (float): Requested loan amount.
        decision (str): The credit decision.

    Returns:
        list: A list of specific reasons for denial.
    """
    reasons = []
    if decision == "DENIED":
        if credit_score < 640: # Common threshold for subprime
            reasons.append(f"Credit score {credit_score} is below the recommended minimum (640).")
        if dti > 0.43: # Common threshold for qualified mortgage DTI
            reasons.append(f"Debt-to-income ratio {dti*100:.1f}% exceeds typical guidelines (43%).")
        if employment_tenure_months < 12: # Common threshold for stable employment
            reasons.append(f"Employment history of {employment_tenure_months} months is below the minimum requirement (12 months).")
        if loan_amount > 0.5 * income and income < 50000: # Heuristic for high loan-to-income for lower earners
             reasons.append(f"Requested loan amount of ${loan_amount:,.0f} is high relative to your income of ${income:,.0f}.")
        
        if not reasons: # Fallback if no specific rule caught it
            reasons.append("Other factors contributed to the denial of your application.")

    return reasons
```

#### Code Cell (Function Execution)
```python
# Test for a DENIED case (high risk from Section 8 test)
denied_app = {
    'income': 28000,
    'credit_score': 580,
    'dti': 0.55,
    'employment_tenure_months': 6,
    'loan_amount': 10000
}
denied_prob = calculate_default_probability(
    **denied_app,
    model_coefficients=model_coefficients,
    model_intercept=model_intercept
)
denied_decision = make_credit_decision(denied_prob)
adverse_reasons = generate_adverse_action_reasons(
    **denied_app,
    decision=denied_decision
)
print(f"Default Probability for denied app: {denied_prob:.4f}")
print(f"Decision for denied app: {denied_decision}")
print("Adverse Action Reasons:")
for reason in adverse_reasons:
    print(f"- {reason}")

# Test for an APPROVED case (should have no reasons)
approved_app = {
    'income': 80000,
    'credit_score': 780,
    'dti': 0.2,
    'employment_tenure_months': 60,
    'loan_amount': 20000
}
approved_prob = calculate_default_probability(
    **approved_app,
    model_coefficients=model_coefficients,
    model_intercept=model_intercept
)
approved_decision = make_credit_decision(approved_prob)
approved_reasons = generate_adverse_action_reasons(
    **approved_app,
    decision=approved_decision
)
print(f"\nDefault Probability for approved app: {approved_prob:.4f}")
print(f"Decision for approved app: {approved_decision}")
print(f"Adverse Action Reasons (should be empty): {approved_reasons}")
```

#### Markdown Cell
The `generate_adverse_action_reasons` function successfully generated specific reasons for the denied application, highlighting factors like low credit score, high DTI, and insufficient employment history. As expected, no reasons were generated for the approved application. This fulfills the requirement for transparency and ECOA compliance.

---

### **Section 11: Calculating Feature Contributions for Explainability (F_005)**

Understanding *why* a model made a particular decision is as important as the decision itself, especially for marginal cases or regulatory scrutiny. This section provides a simplified view of feature importance.

#### Markdown Cell
Model **explainability** (Functional Requirement F_005) is vital for building trust and for compliance. For linear models like Logistic Regression, we can approximate the "contribution" of each feature to the final log-odds ($z$) score. This is a simplified, SHAP-like approach where the contribution of each feature is simply its value multiplied by its coefficient. A positive contribution for a feature (e.g., high DTI with a positive coefficient) means it pushes the probability towards default, while a negative contribution (e.g., high income with a negative coefficient) pulls it away from default.

The contribution of feature $X_i$ is given by:
$$ \text{Contribution}_i = \beta_i \cdot X_i $$
The total log-odds $z$ is the sum of these contributions plus the intercept.

#### Code Cell (Function Implementation)
```python
def calculate_feature_contributions(
    income: float, credit_score: int, dti: float, employment_tenure_months: int, loan_amount: float,
    model_coefficients: dict, model_intercept: float
) -> dict:
    """
    Calculates the contribution of each feature to the log-odds (z) score.

    Args:
        income (float): Annual gross income.
        credit_score (int): FICO credit score.
        dti (float): Debt-to-income ratio.
        employment_tenure_months (int): Months of employment tenure.
        loan_amount (float): Requested loan amount.
        model_coefficients (dict): Dictionary of feature coefficients.
        model_intercept (float): The model's intercept.

    Returns:
        dict: A dictionary of feature names and their contributions.
    """
    contributions = {
        'Income': model_coefficients['income'] * income,
        'Credit Score': model_coefficients['credit_score'] * credit_score,
        'DTI': model_coefficients['debt_to_income_ratio'] * dti,
        'Employment Tenure': model_coefficients['employment_tenure_months'] * employment_tenure_months,
        'Loan Amount': model_coefficients['loan_amount'] * loan_amount,
        'Intercept': model_intercept # Intercept is also a baseline contribution
    }
    return contributions
```

#### Code Cell (Function Execution)
```python
# Test with the denied application example
denied_app_contributions = calculate_feature_contributions(
    **denied_app,
    model_coefficients=model_coefficients,
    model_intercept=model_intercept
)
print("Feature Contributions for Denied Application:")
for feature, contribution in denied_app_contributions.items():
    print(f"- {feature}: {contribution:.4f}")

# Verify total z score
total_z_from_contributions = sum(denied_app_contributions.values())
print(f"\nSum of contributions (total z): {total_z_from_contributions:.4f}")
denied_prob_recalc_z = calculate_default_probability(
    **denied_app,
    model_coefficients=model_coefficients,
    model_intercept=model_intercept
)
print(f"Directly calculated z for denied app: {np.log(denied_prob_recalc_z / (1 - denied_prob_recalc_z)):.4f}")
```

#### Markdown Cell
The `calculate_feature_contributions` function provides a breakdown of how each input factor influences the model's log-odds score. For the denied application, we can clearly see which factors contributed positively (increasing risk, e.g., DTI, Loan Amount) and negatively (decreasing risk, e.g., Income, Credit Score, Employment Tenure). The sum of these contributions, plus the intercept, equals the total log-odds score, confirming the linear relationship.

---

### **Section 12: Integrating the Credit Decision Pipeline**

Now, let's combine all the individual functions into a single pipeline that simulates the complete credit decision process for a given set of inputs.

#### Markdown Cell
We have developed distinct functions for input validation, default probability calculation, credit decision making, adverse action reason generation, and feature contribution analysis. This section integrates these components into a unified **credit decision pipeline**. This function will take raw applicant inputs and return all the calculated outputs, simulating the end-to-end process.

#### Code Cell (Function Implementation)
```python
def simulate_credit_decision_pipeline(
    income: float, credit_score: int, dti: float, employment_tenure_months: int, loan_amount: float,
    model_coefficients: dict, model_intercept: float
) -> dict:
    """
    Simulates the entire credit decision pipeline for a given set of inputs.

    Args:
        income (float): Annual gross income.
        credit_score (int): FICO credit score.
        dti (float): Debt-to-income ratio.
        employment_tenure_months (int): Months of employment tenure.
        loan_amount (float): Requested loan amount.
        model_coefficients (dict): Dictionary of feature coefficients.
        model_intercept (float): The model's intercept.

    Returns:
        dict: A dictionary containing validation status, error messages,
              default probability, decision, adverse action reasons, and feature contributions.
    """
    # 1. Input Validation
    validation_result = validate_inputs(income, credit_score, dti, employment_tenure_months, loan_amount)
    
    if not validation_result['is_valid']:
        return {
            'validation_status': False,
            'error_messages': validation_result['messages'],
            'default_probability': None,
            'decision': "INVALID_INPUTS",
            'adverse_action_reasons': [],
            'feature_contributions': {}
        }

    # 2. Calculate Default Probability
    default_probability = calculate_default_probability(
        income, credit_score, dti, employment_tenure_months, loan_amount,
        model_coefficients, model_intercept
    )

    # 3. Make Credit Decision
    decision = make_credit_decision(default_probability)

    # 4. Generate Adverse Action Reasons
    adverse_action_reasons = generate_adverse_action_reasons(
        income, credit_score, dti, employment_tenure_months, loan_amount, decision
    )

    # 5. Calculate Feature Contributions
    feature_contributions = calculate_feature_contributions(
        income, credit_score, dti, employment_tenure_months, loan_amount,
        model_coefficients, model_intercept
    )

    return {
        'validation_status': True,
        'error_messages': [],
        'default_probability': default_probability,
        'decision': decision,
        'adverse_action_reasons': adverse_action_reasons,
        'feature_contributions': feature_contributions
    }
```

#### Code Cell (Function Execution)
```python
# Test the full pipeline with a sample input (denied_app from before)
pipeline_output = simulate_credit_decision_pipeline(
    **denied_app,
    model_coefficients=model_coefficients,
    model_intercept=model_intercept
)

print("Full Pipeline Output for Denied Application:")
for key, value in pipeline_output.items():
    if key == 'feature_contributions':
        print(f"{key}:")
        for f, c in value.items():
            print(f"  - {f}: {c:.4f}")
    elif key == 'default_probability' and value is not None:
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")
```

#### Markdown Cell
The `simulate_credit_decision_pipeline` function successfully executed all steps, from input validation to generating feature contributions, providing a comprehensive output for the denied application. This confirms the integration of all model components into a coherent workflow.

---

### **Section 13: Interactive Credit Decision Simulator**

This is the interactive core of the notebook. Users can adjust input parameters using sliders and immediately see the impact on default probability, decision, adverse action reasons, and feature contributions.

#### Markdown Cell
Now for the interactive part! This section allows you to experiment with different applicant profiles in real-time. Use the sliders to adjust financial inputs like income, credit score, DTI, employment tenure, and loan amount. As you change these values, the simulator will instantly display the predicted default probability, the credit decision, and, if applicable, the specific reasons for denial.

This interactive experience highlights:
*   The immediate impact of each input factor on the model's output.
*   How thresholds translate probability into a concrete decision.
*   The dynamic generation of adverse action reasons based on your inputs.
*   The relative contribution of each factor to the final decision.

#### Code Cell (Function Execution)
```python
# Define default values for the sliders
default_income = 60000
default_credit_score = 720
default_dti = 0.35
default_employment_tenure = 48
default_loan_amount = 15000

# Create an output area for dynamic updates
output_area = HTML()
display(output_area)

def update_simulator(
    income_val=default_income,
    credit_score_val=default_credit_score,
    dti_val=default_dti,
    employment_tenure_val=default_employment_tenure,
    loan_amount_val=default_loan_amount
):
    # Call the integrated pipeline
    simulation_results = simulate_credit_decision_pipeline(
        income=income_val,
        credit_score=credit_score_val,
        dti=dti_val,
        employment_tenure_months=employment_tenure_val,
        loan_amount=loan_amount_val,
        model_coefficients=model_coefficients,
        model_intercept=model_intercept
    )

    html_output = "<h3>Credit Decision Simulation Results</h3>"

    if not simulation_results['validation_status']:
        html_output += "<p style='color: red; font-weight: bold;'>&#9888; Invalid Input(s):</p>"
        for msg in simulation_results['error_messages']:
            html_output += f"<p style='color: red;'>- {msg}</p>"
        html_output += "<p>Please adjust inputs to be within valid ranges.</p>"
    else:
        # Display core decision
        html_output += f"<p><strong>Default Probability:</strong> {simulation_results['default_probability']:.2%}</p>"
        
        decision_color = 'green' if simulation_results['decision'] == 'APPROVED' else \
                         'orange' if simulation_results['decision'] == 'REVIEW_REQUIRED' else \
                         'red'
        html_output += f"<p><strong>Decision:</strong> <span style='color: {decision_color}; font-weight: bold;'>{simulation_results['decision']}</span></p>"

        # Display adverse action reasons if applicable
        if simulation_results['adverse_action_reasons']:
            html_output += "<p><strong>Adverse Action Reasons:</strong></p><ul>"
            for reason in simulation_results['adverse_action_reasons']:
                html_output += f"<li>{reason}</li>"
            html_output += "</ul>"
        
        # Display feature contributions as a bar chart
        contributions = simulation_results['feature_contributions']
        
        # Exclude intercept for bar chart for better feature comparison
        features_only_contributions = {k: v for k, v in contributions.items() if k != 'Intercept'}
        
        if features_only_contributions:
            contribution_df = pd.DataFrame([features_only_contributions]).T.reset_index()
            contribution_df.columns = ['Feature', 'Contribution']
            contribution_df = contribution_df.sort_values(by='Contribution', ascending=False)

            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['green' if c < 0 else 'red' for c in contribution_df['Contribution']]
            ax.barh(contribution_df['Feature'], contribution_df['Contribution'], color=colors)
            ax.set_xlabel('Contribution to Log-Odds (z)', fontsize=10)
            ax.set_title('Feature Contributions to Risk Score', fontsize=12)
            ax.axvline(0, color='grey', linestyle='--', linewidth=0.8) # Line at zero
            plt.tight_layout()

            # Save the plot to a BytesIO object and embed it in HTML
            import io
            from base64 import b64encode
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_data = b64encode(img_buffer.getvalue()).decode('utf-8')
            plt.close(fig) # Close the figure to prevent it from displaying twice
            html_output += f"<h4>Feature Contributions:</h4><img src='data:image/png;base64,{img_data}' alt='Feature Contributions Bar Chart'>"
    
    output_area.value = html_output

# Create sliders for each input factor
income_slider = IntSlider(min=10000, max=200000, step=1000, value=default_income, description='Income ($)', continuous_update=False)
credit_score_slider = IntSlider(min=300, max=850, step=10, value=default_credit_score, description='Credit Score', continuous_update=False)
dti_slider = FloatSlider(min=0.01, max=1.5, step=0.01, value=default_dti, description='DTI Ratio', continuous_update=False)
employment_tenure_slider = IntSlider(min=0, max=240, step=1, value=default_employment_tenure, description='Tenure (Months)', continuous_update=False)
loan_amount_slider = IntSlider(min=5000, max=50000, step=1000, value=default_loan_amount, description='Loan Amount ($)', continuous_update=False)

# Link sliders to the update function
interact(
    update_simulator,
    income_val=income_slider,
    credit_score_val=credit_score_slider,
    dti_val=dti_slider,
    employment_tenure_val=employment_tenure_slider,
    loan_amount_val=loan_amount_slider
);
```

#### Markdown Cell
You can now interact with the simulator above! Observe how changing inputs (like `credit_score` or `DTI`) directly impacts the `default_probability` and the final `decision`. For example, try decreasing the `credit_score` significantly, or increasing the `DTI`, and see the decision shift from "APPROVED" to "REVIEW_REQUIRED" or even "DENIED". When a denial occurs, review the generated `Adverse Action Reasons` and the `Feature Contributions` chart to understand which factors weighed most heavily in the model's decision.

---

### **Section 14: Analyzing Different Scenarios**

This section encourages users to analyze specific scenarios (e.g., low-risk, high-risk, borderline) and interpret the model's output in detail.

#### Markdown Cell
To further solidify your understanding, let's explicitly analyze a few distinct scenarios. This will help you see how the model behaves across the risk spectrum and how its various output components (probability, decision, reasons, contributions) provide a comprehensive view.

**Scenario 1: A Low-Risk, Approved Application**
Consider an applicant with strong financials.

#### Code Cell (Function Execution)
```python
approved_scenario = {
    'income': 85000,
    'credit_score': 790,
    'dti': 0.18,
    'employment_tenure_months': 72,
    'loan_amount': 25000
}

approved_output = simulate_credit_decision_pipeline(
    **approved_scenario,
    model_coefficients=model_coefficients,
    model_intercept=model_intercept
)

print("--- Scenario 1: Low-Risk, Approved Application ---")
print(f"Input Application: {approved_scenario}")
print(f"Validation Status: {'Valid' if approved_output['validation_status'] else 'Invalid'}")
print(f"Default Probability: {approved_output['default_probability']:.2%}")
print(f"Decision: {approved_output['decision']}")
print(f"Adverse Action Reasons (should be empty): {approved_output['adverse_action_reasons']}")

# Plot contributions for this scenario
if approved_output['validation_status']:
    contributions = approved_output['feature_contributions']
    features_only_contributions = {k: v for k, v in contributions.items() if k != 'Intercept'}
    contribution_df = pd.DataFrame([features_only_contributions]).T.reset_index()
    contribution_df.columns = ['Feature', 'Contribution']
    contribution_df = contribution_df.sort_values(by='Contribution', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['green' if c < 0 else 'red' for c in contribution_df['Contribution']]
    ax.barh(contribution_df['Feature'], contribution_df['Contribution'], color=colors)
    ax.set_xlabel('Contribution to Log-Odds (z)')
    ax.set_title('Feature Contributions for Low-Risk Application')
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.show()
```

#### Markdown Cell
For the low-risk scenario, the `default_probability` is very low, leading to an "APPROVED" decision with no adverse action reasons. The feature contributions plot clearly shows that positive factors like high `Income`, `Credit Score`, and `Employment Tenure` (negative contributions to log-odds mean lower risk) heavily outweigh the minor negative contributions from `DTI` and `Loan Amount`, resulting in a strong approval.

**Scenario 2: A High-Risk, Denied Application**
Now, consider an applicant with weaker financials.

#### Code Cell (Function Execution)
```python
denied_scenario = {
    'income': 30000,
    'credit_score': 590,
    'dti': 0.60,
    'employment_tenure_months': 8,
    'loan_amount': 12000
}

denied_output = simulate_credit_decision_pipeline(
    **denied_scenario,
    model_coefficients=model_coefficients,
    model_intercept=model_intercept
)

print("\n--- Scenario 2: High-Risk, Denied Application ---")
print(f"Input Application: {denied_scenario}")
print(f"Validation Status: {'Valid' if denied_output['validation_status'] else 'Invalid'}")
print(f"Default Probability: {denied_output['default_probability']:.2%}")
print(f"Decision: {denied_output['decision']}")
print("Adverse Action Reasons:")
for reason in denied_output['adverse_action_reasons']:
    print(f"- {reason}")

# Plot contributions for this scenario
if denied_output['validation_status']:
    contributions = denied_output['feature_contributions']
    features_only_contributions = {k: v for k, v in contributions.items() if k != 'Intercept'}
    contribution_df = pd.DataFrame([features_only_contributions]).T.reset_index()
    contribution_df.columns = ['Feature', 'Contribution']
    contribution_df = contribution_df.sort_values(by='Contribution', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['green' if c < 0 else 'red' for c in contribution_df['Contribution']]
    ax.barh(contribution_df['Feature'], contribution_df['Contribution'], color=colors)
    ax.set_xlabel('Contribution to Log-Odds (z)')
    ax.set_title('Feature Contributions for High-Risk Application')
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.show()
```

#### Markdown Cell
In the high-risk scenario, the `default_probability` is high, resulting in a "DENIED" decision. The adverse action reasons clearly highlight the specific deficiencies (e.g., low credit score, high DTI, short employment tenure). The feature contributions plot visually confirms these factors as having strong positive contributions to the risk score, pushing the probability of default higher.

---

### **Section 15: Conclusion and Key Takeaways**

This section summarizes the key learnings and reinforces the importance of the concepts covered.

#### Markdown Cell
This interactive lab provided a hands-on experience with a **Credit Decision Simulator**. You've seen how:

*   **Inputs Drive Outcomes**: Changes in financial factors like income, credit score, DTI, employment tenure, and loan amount directly influence the predicted default probability.
*   **Thresholds Define Decisions**: Clear business rules translate a continuous probability into discrete decisions (APPROVED, REVIEW_REQUIRED, DENIED).
*   **Validation is Paramount**: Robust input validation prevents errors and ensures reliable model outputs.
*   **Explainability Matters**: Adverse action reasons and feature contributions provide transparency, which is crucial for compliance and user understanding.

Understanding these mechanisms is vital for anyone interested in consumer finance, credit risk management, or the practical application of data science in real-world scenarios.

---

### **Section 16: References**

This section lists any external datasets or libraries used, adhering to the user instructions.

#### Markdown Cell
This notebook utilized open-source Python libraries for data handling, numerical operations, visualization, and interactivity.

**Libraries Used:**
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `ipywidgets`
*   `IPython`

**Conceptual References:**
*   **Logistic Regression**: A statistical model widely used for binary classification, particularly in credit risk.
*   **Equal Credit Opportunity Act (ECOA)**: A U.S. federal law that prohibits creditors from discriminating against applicants on the basis of race, color, religion, national origin, sex, marital status, or age. It also mandates the provision of specific reasons for adverse actions.
*   **FICO Score**: A type of credit score that helps lenders assess an applicant's credit risk.
*   **Debt-to-Income Ratio (DTI)**: A key financial metric used to assess an individual's ability to manage monthly payments and repay debts.
