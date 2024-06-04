import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score, accuracy_score

#@st.cache_data(persist=True)
def load_data():
    data=pd.read_csv(r"C:\Users\myohollc\Documents\streamlit\mushrooms\mushrooms.csv")
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data


#@st.cache_data(persist=True)
def split(df):
    y = df.type
    x = df.drop(columns=['type'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, 
                                                        random_state=0)
    return x_train, x_test, y_train, y_test

def plot_metrics(metrics_list, y_test, y_pred):
    class_names=['edible', 'poisonous']
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_pred, display_labels=class_names)
        st.pyplot()
    
    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        RocCurveDisplay.from_predictions(y_true=y_test, y_pred=y_pred)
        st.pyplot()

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        PrecisionRecallDisplay.from_predictions(y_test, y_pred)
        st.pyplot()


def main():
    st.title("Binary Classification Web App")
    st.sidebar.title('Binary Classification Web App')
    st.markdown("Are these mushrooms poisonous?")
    st.sidebar.markdown("Are these mushrooms poisonous?")


    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names=['edible', 'poisonous']


    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df.head())

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ('SVM', 'Logistic Regression', 'Random Forest'))

    if classifier == 'SVM':
        st.sidebar.subheader("Select Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ('rbf', 'linear'), key="kernel")
        gamma = st.sidebar.radio("Gamma (Kernel Coeff)", ("scale", "auto"), key="gamma" )

        metrics = st.sidebar.multiselect("Select metrics to plot", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("SVM Results")
            
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.write("Accuracy", accuracy)
            st.write("Precision", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics, y_test, y_pred)

    # if classifier == 'Logistic Regression':
    #     st.sidebar.subheader("Select Hyperparameters")
    #     C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
    #     kernel = st.sidebar.radio("Kernel", ('rbf', 'linear'), key="kernel")
    #     gamma = st.sidebar.radio("Gamma (Kernel Coeff)", ("scale", "auto"), key="gamma" )

    #     metrics = st.sidebar.multiselect("Select metrics to plot", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

    #     if st.sidebar.button("Classify", key="classify"):
    #         st.subheader("SVM Results")
            
    #         model = SVC(C=C, kernel=kernel, gamma=gamma)
    #         model.fit(x_train, y_train)

    #         y_pred = model.predict(x_test)
    #         accuracy = accuracy_score(y_test, y_pred)

    #         st.write("Accuracy", accuracy)
    #         st.write("Precision", precision_score(y_test, y_pred, labels=class_names).round(2))
    #         st.write("Recall", recall_score(y_test, y_pred, labels=class_names).round(2))
    #         plot_metrics(metrics, y_test, y_pred)




    






if __name__ == '__main__':
    main()


