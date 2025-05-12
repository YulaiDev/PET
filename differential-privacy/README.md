#  Differential Privacy Evaluation using `diffprivlib`

This project evaluates how different machine learning models perform when trained with **differential privacy (DP)** using IBM’s [`diffprivlib`](https://github.com/IBM/differential-privacy-library). All models were trained and tested on the **same dataset**: the [UCI Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult).

---

##  What is `diffprivlib`?

`diffprivlib` is a privacy-preserving machine learning library based on scikit-learn. It ensures individuals' data cannot be inferred even during model training.

 **Key strengths:**

- Easy integration with scikit-learn workflows  
- Adds mathematically calibrated noise to protect data  
- Supports common models:
  - `LogisticRegression`
  - `DecisionTreeClassifier`, `RandomForestClassifier`
  - `GaussianNB`, `LinearRegression`
  - `PCA`, `KMeans`, `StandardScaler`

 **Limitations:**

- No support for deep learning, XGBoost, SVM  
- Manual configuration required for `bounds` and `data_norm`  
- Not suitable for NLP or image-based tasks  

---

## What is Epsilon (𝜖)?

Epsilon (𝜖) is the key parameter in differential privacy:

- **Low 𝜖 (e.g. 0.1)** → Stronger privacy, more noise, lower accuracy  
- **High 𝜖 (e.g. 10)** → Weaker privacy, less noise, higher accuracy  
-  **𝜖 = 1.0** is a commonly accepted balance between privacy and performance

> Think of 𝜖 as a *privacy budget*: the smaller it is, the more privacy you spend per query.

---

##  Model Evaluation: With vs. Without DP (𝜖 = 1.0)

All models were trained on the same dataset. Below are the accuracy comparisons:

| Model               | Without DP | With DP | Accuracy Drop |
|--------------------|------------|---------|----------------|
| Logistic Regression | 81.25%     | 79.50%  | -1.75%         |
| Decision Tree       | 81.86%     | 77.49%  | -4.37%         |

➡ DP was applied with 𝜖 = 1.0 for a fair and realistic comparison.

---

##  Interpretation

- **Logistic Regression** remains very stable with minimal accuracy loss  
- **Decision Trees** are more sensitive due to the effect of noise on split decisions  
- Overall, results are **strong and usable**, even under privacy constraints  
- Confirms that **𝜖 = 1.0 is a reasonable trade-off**

---

##  Linear Regression and R² Score

| Model             | R² Score |
|------------------|----------|
| Non-private       | 0.03     |
| With DP (𝜖 = 1.0) | -0.09    |

- A **negative R²** means the model performs worse than simply predicting the mean of the target  
- ➤ DP seems to work **less well for regression models** than for classification  
- Likely due to higher sensitivity of numeric predictions to added noise  

---

##  Naive Bayes Results

| Model                      | Epsilon | Accuracy |
|---------------------------|---------|----------|
| Non-private GaussianNB    | —       | 79.64%   |
| DP GaussianNB (𝜖 = 1.0)    | 1.0     | 79.93%   |
| DP GaussianNB (𝜖 = 0.1)    | 0.1     | 78.42%   |

- Excellent robustness even at **𝜖 = 0.1**  
- GaussianNB proves effective under strict privacy constraints  
- Accuracy varies slightly with each run due to random noise, as expected with DP  

---

##  Privacy Budgeting and Slack

When applying DP in multiple steps (e.g. scaler + PCA + classifier), your **epsilon budget is consumed cumulatively**.

To manage this:

- Use `BudgetAccountant` to track and control spend  
- Introduce a **`slack`** (e.g. `slack=0.001`) to optimize usage over multiple queries  
  - Allows more operations within the same budget  
  - Accepts a minimal probability of error (delta > 0)  
  - Useful for pipelines or repeated queries  

---

##  Conclusion

- `diffprivlib` is a powerful tool for **privacy-aware machine learning** on structured data  
- **Logistic Regression** performs best under DP  
- **Decision Trees** remain usable but are more impacted by noise  
- **GaussianNB** performs surprisingly well, even at low epsilon  
- **Linear Regression** under DP yields weaker results — better suited for classification tasks  
- **𝜖 = 1.0** remains a practical and responsible default setting  

---

