
```markdown
# Heart Disease Prediction & Analysis System 🫀

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 📌 About The Project
This project was developed as part of an advanced data science training program in collaboration with **Sprite & Microsoft**. It is an end-to-end Machine Learning pipeline designed to predict the risk of heart disease based on clinical parameters. 

The project goes beyond simple classification by incorporating comprehensive data preprocessing, hyperparameter tuning, unsupervised learning for data exploration, and an interactive Arabic web application for real-time risk assessment.

## 🚀 Key Features
- **Robust ML Pipeline:** Utilizes `scikit-learn` Pipelines with `ColumnTransformer` to handle both numerical scaling (`StandardScaler`) and categorical encoding (`OneHotEncoder`) seamlessly.
- **Model Evaluation:** Evaluates multiple algorithms (Logistic Regression, Decision Tree, Random Forest, SVC) using Stratified K-Fold Cross Validation.
- **Hyperparameter Tuning:** Implements `GridSearchCV` to find the optimal parameters for the Random Forest Classifier.
- **Unsupervised Learning:** Explores natural data groupings using KMeans Clustering, PCA (Principal Component Analysis) for dimensionality reduction, and Hierarchical Clustering dendrograms.
- **Interactive UI:** Features a fully functional web interface built with **Streamlit** (in Arabic), allowing users to input medical parameters, adjust decision thresholds, and view prediction probabilities interactively.

## 📂 Repository Structure
```text
Heart_Disease_Project/
│
├── data/                   # Contains the heart.csv dataset
├── src/                    # Source code for model training scripts
├── ui/                     # Streamlit web application files
├── models/                 # Saved models (final_model.pkl) and metadata
├── results/                # Evaluation metrics, PCA scatters, and CSV reports
├── finalize_project.py     # Main script to run CV, grid search, and generate artifacts
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

```

## 🛠️ Installation & Setup

**1. Clone the repository:**

```bash
git clone [https://github.com/yahyalababenah/Heart_Disease_Project.git](https://github.com/yahyalababenah/Heart_Disease_Project.git)
cd Heart_Disease_Project

```

**2. Install dependencies:**
It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt

```

**3. Run the ML Pipeline:**
To train the model, perform cross-validation, and generate all clustering visualizations and metrics, run:

```bash
python finalize_project.py

```

*(Check the `results/` folder for generated artifacts like `supervised_metrics.csv` and `pca_scatter.png`)*

**4. Launch the Web Application:**
To start the Streamlit interactive UI:

```bash
streamlit run ui/app.py 

```

*(Note: Replace `app.py` with the exact name of your python file inside the `ui` folder)*

## 📊 Data Source

The dataset used in this project features clinical attributes such as age, sex, chest pain type (cp), resting blood pressure (trestbps), cholesterol levels (chol), and ECG results to predict the presence of heart disease (`target` variable). Provided via the Sprite & Microsoft training program.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

```

