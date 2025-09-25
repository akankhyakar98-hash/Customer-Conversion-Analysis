import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import pandas as pd

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "ML Tasks",
        ["Classification", "Regression", "Clustering"],
        icons=["bar-chart", "graph-up", "diagram-3"],
        menu_icon="cast",
        default_index=0
    )

# CLASSIFICATION MODULE
if selected == "Classification":
    st.title("Purchase Prediction App")
    st.markdown("Predict whether a user will make a purchase based on page and product features.")

    # Load the saved classification pipeline
    try:
        with open('xgboost_classifier_pipeline.pkl', 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        st.error(" Model file not found. Please make sure 'xgboost_classifier_pipeline.pkl' is in the app directory.")
    else:
        # User inputs
        st.subheader(" Enter Feature Values")

        country = st.selectbox("Country", ['India', 'USA', 'UK', 'Germany', 'France'])
        page1_main_category = st.selectbox("Main Category", ['Men', 'Women', 'Kids', 'Accessories'])
        page2_clothing_model = st.selectbox("Clothing Model", ['T-Shirt', 'Jeans', 'Dress', 'Shoes'])
        colour = st.selectbox("Colour", ['Red', 'Blue', 'Green', 'Black', 'White'])
        location = st.selectbox("Location", ['Homepage', 'Product Page', 'Checkout'])
        model_photography = st.selectbox("Model Photography", ['Studio', 'Outdoor', 'None'])

        order = st.number_input("Order Count", min_value=0, value=1)
        revenue = st.number_input("Revenue", min_value=0.0, value=100.0)
        page = st.number_input("Page Number", min_value=1, value=1)

        year = st.selectbox("Year", [2022, 2023, 2024, 2025])
        month = st.selectbox("Month", list(range(1, 13)))
        day = st.selectbox("Day", list(range(1, 32)))

        # Predict button
        if st.button(" Predict Purchase"):
            input_df = pd.DataFrame([{
                'year': year,
                'month': month,
                'day': day,
                'order': order,
                'country': country,
                'page1_main_category': page1_main_category,
                'page2_clothing_model': page2_clothing_model,
                'colour': colour,
                'location': location,
                'model_photography': model_photography,
                'page': page,
                'revenue': revenue
            }])

            try:
                prediction = model.predict(input_df)
                result = " Will Purchase" if prediction[0] == 1 else " Will Not Purchase"
                st.success(f"Prediction: {result}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

                st.write(" Input DataFrame:", input_df)

# # REGRESSION MODULE

elif selected == "Regression":
    st.title(" Revenue Prediction App")
    st.markdown("Estimate expected revenue for a user based on their behavior and attributes.")

    try:
        with open('revenue_regression_pipeline.pkl', 'rb') as file:
            reg_model = pickle.load(file)
    except FileNotFoundError:
        st.error(" Regression model file not found. Please make sure 'revenue_regression_pipeline.pkl' is in the app directory.")
    else:
        st.subheader(" Enter Feature Values")

        # Categorical inputs
        country = st.selectbox("Country", ['India', 'USA', 'UK', 'Germany', 'France'])
        page1_main_category = st.selectbox("Main Category", ['Men', 'Women', 'Kids', 'Accessories'])
        page2_clothing_model = st.selectbox("Clothing Model", ['T-Shirt', 'Jeans', 'Dress', 'Shoes'])
        colour = st.selectbox("Colour", ['Red', 'Blue', 'Green', 'Black', 'White'])
        location = st.selectbox("Location", ['Homepage', 'Product Page', 'Checkout'])
        model_photography = st.selectbox("Model Photography", ['Studio', 'Outdoor', 'None'])

        # Numerical inputs
        order = st.number_input("Order Count", min_value=0, value=1)
        page = st.number_input("Page Number", min_value=1, value=1)
        revenue = st.number_input("Revenue", min_value=0.0, value=100.0)  #  Add this to match training features

        # Date inputs
        year = st.selectbox("Year", [2022, 2023, 2024, 2025])
        month = st.selectbox("Month", list(range(1, 13)))
        day = st.selectbox("Day", list(range(1, 32)))

        if st.button("Predict Revenue"):
            import pandas as pd

            input_df = pd.DataFrame([{
                'year': year,
                'month': month,
                'day': day,
                'order': order,
                'country': country,
                'page1_main_category': page1_main_category,
                'page2_clothing_model': page2_clothing_model,
                'colour': colour,
                'location': location,
                'model_photography': model_photography,
                'page': page,
                 
            }])

            try:
                predicted_revenue = reg_model.predict(input_df)[0]
                st.success(f" Predicted Revenue: ₹{predicted_revenue:.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.write(" Input DataFrame:", input_df)
# clustering
elif selected == "Clustering":
    st.title(" User Segmentation App")
    st.markdown("Segment users based on browsing and product interaction features to enable personalized marketing and product recommendations.")

    try:
        with open('kmeans_model.pkl', 'rb') as f:
            cluster_model = pickle.load(f)
        
        with open('scalar2.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        st.error(" Clustering model or scaler not found. Please ensure 'dbscan_model.pkl' and 'scaler.pkl' are in the app directory.")
    else:
        st.subheader(" Enter User Behavior Features")

        # Inputs for clustering features
        order = st.number_input("Order Count", min_value=0, value=1)
        page = st.number_input("Page Number", min_value=1, value=1)
        country = st.selectbox("Country", ['India', 'USA', 'UK', 'Germany', 'France'])
        page1_main_category = st.selectbox("Main Category", ['Men', 'Women', 'Kids', 'Accessories'])
        page2_clothing_model = st.selectbox("Clothing Model", ['T-Shirt', 'Jeans', 'Dress', 'Shoes'])
        colour = st.selectbox("Colour", ['Red', 'Blue', 'Green', 'Black', 'White'])
        location = st.selectbox("Location", ['Homepage', 'Product Page', 'Checkout'])
        model_photography = st.selectbox("Model Photography", ['Studio', 'Outdoor', 'None'])

        # Mapping categorical values to match training
        country_map = {'India': 0, 'USA': 1, 'UK': 2, 'Germany': 3, 'France': 4}
        colour_map = {'Red': 0, 'Blue': 1, 'Green': 2, 'Black': 3, 'White': 4}
        location_map = {'Homepage': 0, 'Product Page': 1, 'Checkout': 2}
        model_photography_map = {'Studio': 0, 'Outdoor': 1, 'None': 2}
        main_category_map = {'Men': 0, 'Women': 1, 'Kids': 2, 'Accessories': 3}
        clothing_model_map = {'T-Shirt': 0, 'Jeans': 1, 'Dress': 2, 'Shoes': 3}

        # Create input DataFrame with encoded values
        input_df = pd.DataFrame([{
            'order': order,
            'page': page,
            'country': country_map[country],
            'page1_main_category': main_category_map[page1_main_category],
            'page2_clothing_model': clothing_model_map[page2_clothing_model],
            'colour': colour_map[colour],
            'location': location_map[location],
            'model_photography': model_photography_map[model_photography]
        }])

        if st.button(" Find Cluster"):
            try:
                # Scale input
                scaled_input = scaler.transform(input_df)

                # Predict cluster label
                cluster_label = cluster_model.predict(scaled_input)[0]

                if cluster_label == -1:
                    st.warning(" This user is considered noise (not part of any cluster).")
                else:
                    st.success(f" User belongs to Cluster: {cluster_label}")
            except Exception as e:
                st.error(" Clustering failed due to an error.")
                st.write("Error details:", e)
                st.write("Input DataFrame:", input_df)
    