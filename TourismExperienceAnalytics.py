# Data Preprocessing

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from surprise import Dataset, Reader, SVD
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise import accuracy

# Load datasets (replace with actual file paths)
transaction_data = pd.read_excel('C:\\ML\\TourismExperience\\Tourism Dataset\\Transaction.xlsx')
user_data = pd.read_excel('C:\\ML\\TourismExperience\\Tourism Dataset\\user.xlsx')
city_data = pd.read_excel('C:\\ML\\TourismExperience\\Tourism Dataset\\City.xlsx')
Continent_data = pd.read_excel('C:\\ML\\TourismExperience\\Tourism Dataset\\Continent.xlsx')
Item_data = pd.read_excel('C:\\ML\\TourismExperience\\Tourism Dataset\\Item.xlsx')
Region_data = pd.read_excel('C:\\ML\\TourismExperience\\Tourism Dataset\\Region.xlsx')
Type_data = pd.read_excel('C:\\ML\\TourismExperience\\Tourism Dataset\\Type.xlsx')
Mode_data = pd.read_excel('C:\\ML\\TourismExperience\\Tourism Dataset\\Mode.xlsx')
Country_data = pd.read_excel('C:\\ML\\TourismExperience\\Tourism Dataset\\Country.xlsx')

user_data['CityId'] = user_data['CityId'].fillna(user_data['CityId'].mean())

merged_data = pd.merge(transaction_data, user_data, on='UserId', how='left')
country_region_merged_data = pd.merge(Country_data, Region_data, on='RegionId', how='left')
merged_data = pd.merge(merged_data, Continent_data, on='ContenentId', how='left')
merged_data = pd.merge(merged_data, country_region_merged_data, on='RegionId', how='left')
merged_data = pd.merge(merged_data, Item_data, on='AttractionId', how='left')
merged_data = pd.merge(merged_data, Type_data, on='AttractionTypeId', how='left')
merged_data['VisitModeId'] = merged_data['VisitMode']
merged_data = pd.merge(merged_data, Mode_data, on='VisitModeId', how='left')
print('merged_data:\n',merged_data)
merged_data.to_csv('C:\\ML\\TourismExperience\\Tourism Dataset\\Merged_TorismExperince.csv')
cleaned_data = pd.read_csv('C:\\ML\\TourismExperience\\Tourism Dataset\\Merged_TorismExperince.csv') 

print(cleaned_data.isnull().sum())
# Set plot style
sns.set(style="whitegrid", palette="pastel")

# Step 1: Distribution of Visit Modes
plt.figure(figsize=(8, 5))
sns.countplot(data=cleaned_data, x="VisitMode_y", order=cleaned_data["VisitMode_y"].value_counts().index)
plt.title("Distribution of Visit Modes")
plt.xlabel("Visit Mode")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Step 2: Distribution of Attraction Types
plt.figure(figsize=(12, 6))
sns.countplot(data=cleaned_data, y="AttractionType", order=cleaned_data["AttractionType"].value_counts().index)
plt.title("Distribution of Attraction Types")
plt.xlabel("Count")
plt.ylabel("Attraction Type")
plt.show()


# Corrected code for barplot with Seaborn
plt.figure(figsize=(8, 5))

continent_rating = cleaned_data.groupby("Contenent")["Rating"].mean().sort_values()
# Create a DataFrame for better compatibility with seaborn
continent_rating_df = pd.DataFrame({
    "Continent": continent_rating.index,
    "Average Rating": continent_rating.values
})

sns.barplot(data=continent_rating_df, x="Continent", y="Average Rating", hue="Continent", dodge=False, palette="viridis", legend=False)
plt.title("Average Rating by Continent")
plt.xlabel("Continent")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.show()




plt.figure(figsize=(10, 8))
# Select only numerical columns
numerical_columns = cleaned_data.select_dtypes(include=["int64", "float64"])
correlation_matrix = numerical_columns.corr()

# Plot the heatmap
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap for Numerical Features")
plt.show()


def Regression_Task():
        # Ensure cleaned_data is the dataset after preprocessing
        # Replace this with your prepared dataset if necessary
        df = cleaned_data.copy()

        # Step 1: Encode Categorical Variables
        df_encoded = pd.get_dummies(df, columns=['Contenent', 'Region', 'Country', 'VisitMode_y', 'AttractionType'], drop_first=True)

        # Step 2: Define Features (X) and Target (y)
        X = df_encoded.drop(columns=['Rating', 'Attraction', 'AttractionAddress'])
        y = df_encoded['Rating']

        # Step 3: Split Data into Training and Testing Sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 4: Initialize Models
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "XGBoost": XGBRegressor(random_state=42)
        }

        # Step 5: Train and Evaluate Models
        results = {}
        for model_name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            results[model_name] = {"RMSE": rmse, "MAE": mae, "R²": r2}

        # Step 6: Display Results
        print("Model Performance:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")


#Code for Classification Task

def ClassificationTask():
    df = cleaned_data.copy()

    #Encode Categorical Variables
    df_encoded = pd.get_dummies(df, columns=['Contenent', 'Region', 'Country', 'AttractionType'], drop_first=True)

    # Clean column names by removing special characters and spaces
    df_encoded.columns = df_encoded.columns.str.replace(r'\W+', '_', regex=True)  # Replace non-alphanumeric characters with underscores

    # Re-initialize and train the model (LightGBM example)
    X = df_encoded.drop(columns=['VisitMode_y', 'Rating', 'Attraction', 'AttractionAddress'])
    y = df_encoded['VisitMode_y']

    # Encode target labels (VisitMode_y) as integers
    y = y.astype('category').cat.codes

    #Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    #Initialize Classifiers
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "LightGBM": LGBMClassifier(random_state=42)
    }

    #Train and Evaluate Models
    results = {}
    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store results
        results[model_name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Classification Report": classification_report(y_test, y_pred)
        }

    #Display Results
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            if metric_name == "Classification Report":
                print(f"\n{metric_name}:\n{value}")
            else:
                print(f"{metric_name}: {value:.4f}")


#Code for Recommendation System
def RecommendationSystem():
    #Prepare Data for Surprise Library
    # Convert dataset to Surprise-compatible format (user-item-rating)
    reader = Reader(rating_scale=(1, 5))
    recommendation_data = cleaned_data[['UserId', 'AttractionId', 'Rating']]
    data = Dataset.load_from_df(recommendation_data, reader)

    # Step 2: Train-Test Split
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Step 3: Collaborative Filtering Models
    # a) User-based Collaborative Filtering
    user_based_model = KNNBasic(sim_options={'name': 'pearson_baseline', 'user_based': True})
    user_based_model.fit(trainset)

    # b) Item-based Collaborative Filtering
    item_based_model = KNNBasic(sim_options={'name': 'pearson_baseline', 'user_based': False})
    item_based_model.fit(trainset)

    # c) Matrix Factorization (SVD)
    svd_model = SVD()
    svd_model.fit(trainset)

    # Step 4: Evaluate Models
    print("Evaluation Results:")
    for model_name, model in [("User-Based CF", user_based_model), 
                            ("Item-Based CF", item_based_model), 
                            ("SVD", svd_model)]:
        predictions = model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        print(f"{model_name}: RMSE = {rmse:.4f}")

    # Step 5: Generate Recommendations (Top-N)
    def get_top_n(predictions, n=5):
        """Return top-N recommendations for each user from a set of predictions."""
        from collections import defaultdict
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Sort predictions for each user and retrieve the top N
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]
        return top_n

    # Generate top 5 recommendations for each user using the best-performing model
    predictions = svd_model.test(testset)
    top_n_recommendations = get_top_n(predictions, n=5)

    # Display Recommendations for a Sample User
    sample_user = list(top_n_recommendations.keys())[0]
    print(f"\nTop 5 Recommendations for User {sample_user}:")
    for attraction_id, estimated_rating in top_n_recommendations[sample_user]:
        print(f"Attraction ID: {attraction_id}, Estimated Rating: {estimated_rating:.2f}")



def StreamlitApp():
    # Set Streamlit page configuration
    st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")

    # Load Data
    @st.cache_data
    def load_data():
        cleaned_data = pd.read_csv("C:\\ML\\TourismExperience\\Tourism Dataset\\Merged_TorismExperince.csv")
        return cleaned_data

    data = load_data()

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Go to", ["Overview", "Regression", "Classification", "Recommendations"])

    # --- Helper Functions ---
    def create_visualizations(data):
        # Visit Modes Distribution with Filter
        st.subheader("Distribution of Visit Modes")
        region_filter = st.selectbox("Filter by Region", ["All"] + list(data["Region"].unique()))
        filtered_data = data if region_filter == "All" else data[data["Region"] == region_filter]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=filtered_data, x="VisitMode_y", order=filtered_data["VisitMode_y"].value_counts().index, palette="Set2", ax=ax)
        plt.xticks(rotation=45)
        plt.title(f"Visit Mode Distribution ({region_filter})")
        st.pyplot(fig)

        # Dynamic Attraction Type Chart
        st.subheader("Attraction Types by Region")
        attraction_filter = st.multiselect("Select Attraction Types", data["AttractionType"].unique(), default=data["AttractionType"].unique())
        filtered_data = data[data["AttractionType"].isin(attraction_filter)]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=filtered_data, y="AttractionType", order=filtered_data["AttractionType"].value_counts().index, palette="coolwarm", ax=ax)
        plt.title(f"Attraction Types ({', '.join(attraction_filter)})")
        st.pyplot(fig)

        # Average Rating by Continent with Sorting
        st.subheader("Average Rating by Continent")
        sort_by_rating = st.checkbox("Sort by Rating", value=True)
        continent_rating = data.groupby("Contenent")["Rating"].mean().sort_values(ascending=sort_by_rating)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=continent_rating.index, y=continent_rating.values, palette="viridis", ax=ax)
        ax.set_title("Average Rating by Continent")
        st.pyplot(fig)

    # --- Overview ---
    if options == "Overview":
        st.title("Tourism Experience Analytics Dashboard")
        st.write("Explore insights, make predictions, and generate recommendations for tourism experiences.")
        create_visualizations(data)

    # --- Regression ---
    elif options == "Regression":
        st.title("Attraction Rating Prediction (Regression)")
        
        # Prepare Data
        df_encoded = pd.get_dummies(data, columns=['Contenent', 'Region', 'Country', 'VisitMode_y', 'AttractionType'], drop_first=True)
        X = df_encoded.drop(columns=['Rating', 'Attraction', 'AttractionAddress'])
        y = df_encoded['Rating']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # User Input
        st.sidebar.subheader("Input Features for Regression")
        user_inputs = {col: st.sidebar.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean())) for col in X.columns}
        user_input_df = pd.DataFrame([user_inputs])

        st.subheader("Your Input Features")
        st.write(user_input_df)

        # Predict Rating
        if st.sidebar.button("Predict Rating"):
            prediction = model.predict(user_input_df)
            st.subheader("Predicted Attraction Rating")
            st.write(f"Predicted Rating: {prediction[0]:.2f}")

        # Visualize Feature Importances
        st.subheader("Feature Importances")
        importances = model.feature_importances_
        importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax, palette="mako")
        plt.title("Feature Importance for Regression Model")
        st.pyplot(fig)

    # --- Classification ---
    elif options == "Classification":
        st.title("Visit Mode Prediction (Classification)")
        
        # Prepare Data
        df_encoded = pd.get_dummies(data, columns=['Contenent', 'Region', 'Country', 'AttractionType'], drop_first=True)
        X = df_encoded.drop(columns=['VisitMode_y', 'Rating', 'Attraction', 'AttractionAddress'])
        y = data['VisitMode_y'].astype('category').cat.codes
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        classifier = RandomForestClassifier(random_state=42)
        classifier.fit(X_train, y_train)

        # User Input
        st.sidebar.subheader("Input Features for Classification")
        user_inputs = {col: st.sidebar.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean())) for col in X.columns}
        user_input_df = pd.DataFrame([user_inputs])

        st.subheader("Your Input Features")
        st.write(user_input_df)

        # Predict Visit Mode
        if st.sidebar.button("Predict Visit Mode"):
            prediction = classifier.predict(user_input_df)
            visit_mode = data['VisitMode_y'].astype('category').cat.categories[prediction[0]]
            st.subheader("Predicted Visit Mode")
            st.write(f"Predicted Visit Mode: {visit_mode}")

        # Display Classification Report
        y_pred = classifier.predict(X_test)
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

    # --- Recommendations ---
    elif options == "Recommendations":
        st.title("Attraction Recommendations")

        # Prepare Recommendation Data
        reader = Reader(rating_scale=(1, 5))
        recommendation_data = data[['UserId', 'AttractionId', 'Rating']]
        surprise_data = Dataset.load_from_df(recommendation_data, reader)
        trainset = surprise_data.build_full_trainset()
        svd = SVD()
        svd.fit(trainset)

        # User Input
        user_id = st.sidebar.number_input("Enter User ID", min_value=int(data['UserId'].min()), max_value=int(data['UserId'].max()))
        if st.sidebar.button("Get Recommendations"):
            all_attractions = data['AttractionId'].unique()
            user_ratings = [(attraction, svd.predict(user_id, attraction).est) for attraction in all_attractions]
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            recommendations = user_ratings[:5]

            st.subheader(f"Top 5 Recommendations for User {user_id}")
            for attraction_id, est_rating in recommendations:
                attraction_name = data[data['AttractionId'] == attraction_id]['Attraction'].iloc[0]
                st.write(f"Attraction: {attraction_name}, Estimated Rating: {est_rating:.2f}")

        if __name__ == "__main__" :           
            StreamlitApp()
