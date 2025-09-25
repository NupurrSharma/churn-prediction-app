import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="Churn Insights & Predictor", layout="wide")

# --- Load Model and Data ---
@st.cache_data
def load_data_and_model():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    model = joblib.load('churn_predictor_model.joblib')
    model_columns = joblib.load('model_columns.joblib')
    return df, model, model_columns

df, model, model_columns = load_data_and_model()

# --- Helper Function for Charts ---
def create_churn_bar_chart(df, column, title):
    churn_data = df.groupby(column)['Churn'].value_counts(normalize=True).unstack().mul(100)
    churn_data = churn_data.rename(columns={'No': 'Not Churned', 'Yes': 'Churned'})
    
    fig = px.bar(churn_data,
                 title=title,
                 labels={'value': 'Percentage of Customers (%)', column: column},
                 text_auto='.2f',
                 color_discrete_map={'Not Churned': '#1f77b4', 'Churned': '#d62728'}) # Blue for stay, Red for churn
    fig.update_layout(title_x=0.5, yaxis_title="Percentage of Customers (%)")
    return fig

# --- App UI ---
st.title('Customer Churn Insights & Predictor 🔮')

# --- Prediction Tool (collapsible) ---
with st.expander("🤖 Real-Time Churn Predictor"):
    st.sidebar.header('Customer Details for Prediction')
    def user_input_features():
        tenure = st.sidebar.slider('Tenure (Months)', 0, 72, 24)
        monthly_charges = st.sidebar.slider('Monthly Charges ($)', 18.0, 120.0, 70.0)
        total_charges = st.sidebar.slider('Total Charges ($)', 18.0, 9000.0, 1500.0)
        contract = st.sidebar.selectbox('Contract Type', sorted(df['Contract'].unique()))
        tech_support = st.sidebar.selectbox('Tech Support', sorted(df['TechSupport'].unique()))
        online_security = st.sidebar.selectbox('Online Security', sorted(df['OnlineSecurity'].unique()))
        payment_method = st.sidebar.selectbox('Payment Method', sorted(df['PaymentMethod'].unique()))
        internet_service = st.sidebar.selectbox('Internet Service', sorted(df['InternetService'].unique()))
        data = {'tenure': tenure, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges, 'Contract': contract, 'TechSupport': tech_support, 'OnlineSecurity': online_security, 'PaymentMethod': payment_method, 'InternetService': internet_service}
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()
    encoded_df = pd.get_dummies(input_df)
    query = encoded_df.reindex(columns=model_columns, fill_value=0)
    prediction_proba = model.predict_proba(query)[:, 1][0]
    
    st.metric(label="Churn Probability", value=f"{prediction_proba:.2%}")
    if prediction_proba > 0.5: st.error('🔴 High Risk of Churn')
    else: st.success('🟢 Low Risk of Churn')

st.markdown("---")

# --- Dynamic Dashboard ---
st.header('Interactive Churn Dashboard')
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "👥 Demographics", "🔧 Services & Contracts", "💳 Financials"])

# --- Tab 1: Overview ---
with tab1:
    churn_rate = (df['Churn'] == 'Yes').mean()
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Churn Rate", f"{churn_rate:.2%}")
    col2.metric("Total Customers", f"{df.shape[0]}")
    col3.metric("Total Churned", f"{(df['Churn'] == 'Yes').sum()}")
    
    churn_dist = df['Churn'].value_counts()
    fig_donut = px.pie(churn_dist, values=churn_dist.values, names=churn_dist.index, title='Overall Churn Distribution', hole=0.4, color_discrete_map={'No':'#1f77b4', 'Yes':'#d62728'})
    st.plotly_chart(fig_donut, use_container_width=True)

# --- Tab 2: Demographics ---
with tab2:
    st.subheader("How Demographics and Family Structure Affect Churn")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_churn_bar_chart(df, 'gender', 'Churn Rate by Gender'), use_container_width=True)
        st.markdown("**Insight:** Gender has a negligible impact on churn.")
        st.plotly_chart(create_churn_bar_chart(df, 'Partner', 'Churn Rate by Partner Status'), use_container_width=True)
        st.markdown("**Insight:** Customers without a partner are significantly more likely to churn.")
    with col2:
        df['SeniorCitizen_str'] = df['SeniorCitizen'].map({0: 'Not Senior', 1: 'Senior'})
        st.plotly_chart(create_churn_bar_chart(df, 'SeniorCitizen_str', 'Churn Rate by Senior Citizen Status'), use_container_width=True)
        st.markdown("**Insight:** Senior citizens exhibit a much higher churn rate than younger customers.")
        st.plotly_chart(create_churn_bar_chart(df, 'Dependents', 'Churn Rate by Dependents'), use_container_width=True)
        st.markdown("**Insight:** Customers with no dependents have a higher tendency to churn.")

# --- Tab 3: Services & Contracts ---
with tab3:
    st.subheader("How Services and Contract Terms Influence Churn")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_churn_bar_chart(df, 'Contract', 'Churn Rate by Contract Type'), use_container_width=True)
        st.markdown("**Insight:** Month-to-month contracts have an overwhelmingly high churn rate compared to long-term contracts.")
        st.plotly_chart(create_churn_bar_chart(df, 'TechSupport', 'Churn Rate by Tech Support Subscription'), use_container_width=True)
        st.markdown("**Insight:** Lack of tech support is a major driver of churn.")
    with col2:
        st.plotly_chart(create_churn_bar_chart(df, 'InternetService', 'Churn Rate by Internet Service'), use_container_width=True)
        st.markdown("**Insight:** Customers with Fiber optic internet churn more frequently, possibly due to higher costs or service issues.")
        st.plotly_chart(create_churn_bar_chart(df, 'OnlineSecurity', 'Churn Rate by Online Security Subscription'), use_container_width=True)
        st.markdown("**Insight:** Customers without online security are more than twice as likely to churn.")

# --- Tab 4: Financials ---
with tab4:
    st.subheader("How Financial Factors Relate to Churn")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_churn_bar_chart(df, 'PaymentMethod', 'Churn Rate by Payment Method'), use_container_width=True)
        st.markdown("**Insight:** Customers paying by electronic check churn at a much higher rate, indicating potential dissatisfaction with this payment process.")
    with col2:
        fig_hist = px.histogram(df, x='MonthlyCharges', color='Churn', marginal='box', title='Distribution of Monthly Charges by Churn',
                                color_discrete_map={'No':'#1f77b4', 'Yes':'#d62728'})
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown("**Insight:** Churn is more prevalent among customers with higher monthly charges (approx. $70-$100).")
