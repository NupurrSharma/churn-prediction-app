# Customer Churn Prediction & Analytics Dashboard

An end-to-end data science project that analyzes telecom customer data to predict churn, explain the predictions, and quantify the financial impact. This project is deployed as a full-stack, interactive web application using Streamlit.

### 🔴 [Live Application Demo]([https://YOUR-APP-URL.streamlit.app/](https://churn-prediction-app-jsufzptaj65wznybrq75kv.streamlit.app/))






<img width="1423" height="799" alt="image" src="https://github.com/user-attachments/assets/a77123f2-140d-4098-852d-d10da95c1cbd" />
<img width="1423" height="799" alt="image" src="https://github.com/user-attachments/assets/ae61b193-7e8b-49ce-b8f0-ec105ac839f3" />



## Project Overview

This project tackles the critical business problem of customer churn. Instead of a static analysis, this tool provides a comprehensive, interactive dashboard for exploring historical data and a real-time prediction engine for identifying at-risk customers.

What makes this project unique is its focus on the entire data science lifecycle:
1.  **Analysis:** Understanding the "why" behind churn through a multi-tabbed, interactive dashboard.
2.  **Prediction:** Using a trained XGBoost model to predict "who" is likely to churn.
3.  **Explanation (XAI):** Integrating SHAP to explain "why" the model makes a specific prediction for an individual.
4.  **Business Impact:** Quantifying the "so what?" by calculating Customer Lifetime Value (CLV) and Revenue at Risk.

## Key Features

* **📈 Interactive Dashboard:** A multi-tab dashboard built with Plotly to explore churn drivers across demographics, services, contracts, and financials.
* **🤖 Real-Time Predictor:** An interactive sidebar where users can input customer details and instantly receive a churn probability score from the live XGBoost model.
* **🧠 Explainable AI (XAI):** For every prediction, a SHAP force plot is generated to show which features are pushing the churn risk up or down, making the model's decisions transparent.
* **💰 Financial Impact Analysis:** The application calculates the estimated CLV for the customer profile and the specific "Revenue at Risk" to translate data insights into business value.

## Tech Stack

* **Language:** Python
* **Libraries:** Pandas, Scikit-learn, XGBoost, SHAP, Plotly, Streamlit
* **Deployment:** Streamlit Community Cloud, GitHub

## How to Run Locally

1.  Clone the repository:
    ```sh
    git clone [https://github.com/YourUsername/churn-prediction-app.git](https://github.com/YourUsername/churn-prediction-app.git)
    ```
2.  Navigate to the project directory:
    ```sh
    cd churn-prediction-app
    ```
3.  Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```
4.  Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

---
