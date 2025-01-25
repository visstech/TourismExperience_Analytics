# Tourism Experience Analytics Dashboard

## Overview

The **Tourism Experience Analytics Dashboard** is a machine learning-based Streamlit application that provides:

1. **Insights into Tourism Data**:
   - Distribution of visit modes and attraction types.
   - Average ratings by continent and other patterns.
2. **Machine Learning Models**:
   - Regression for predicting attraction ratings.
   - Classification for predicting visit modes.
   - Collaborative filtering-based recommendation system for attractions.
3. **Interactive Visualizations**:
   - Dynamic filters for better exploration of the data.
   - Intuitive graphs and charts for quick insights.

---

## Features

### 1. **Data Insights and Visualizations**

- **Visit Modes Distribution**: Visualize the distribution of visit modes with options to filter by region.
- **Attraction Types by Region**: Explore attraction types dynamically by selecting specific types.
- **Average Ratings**: See the average attraction ratings by continent, with sorting options for better insights.

### 2. **Regression (Attraction Rating Prediction)**

- **Objective**: Predict the rating of a tourist attraction based on features like user demographics, attraction type, and visit details.
- **Model**: Random Forest Regressor.
- **Interactive Inputs**: Users can adjust sliders for feature inputs and get a predicted rating.
- **Feature Importances**: Visualize the key drivers for the regression model.

### 3. **Classification (Visit Mode Prediction)**

- **Objective**: Predict the visit mode (e.g., Business, Family, Friends) based on user and attraction data.
- **Model**: Random Forest Classifier.
- **Interactive Inputs**: Input custom features and get the predicted visit mode.
- **Performance Metrics**: Displays a detailed classification report.

### 4. **Recommendations**

- **Objective**: Recommend the top 5 attractions for a specific user based on their history and preferences.
- **Model**: Collaborative filtering (SVD from Surprise library).
- **Interactive Inputs**: Users can input their User ID to receive personalized recommendations.

---

## Installation and Setup

### Prerequisites

Ensure you have Python 3.7+ installed on your system. Install the required libraries:

```bash
pip install streamlit pandas seaborn matplotlib scikit-learn surprise
```

### Running the Application

1. Clone the repository or copy the project files to your local machine.
2. Navigate to the project directory and run:
   ```bash
   streamlit run tourism_app.py
   ```
3. Open the provided URL in your browser to access the application.

---

## Project Structure

```
├── tourism_app.py         # Main Streamlit application
├── data
│   └── Merged_TourismExperience.csv   # Preprocessed dataset
├── README.md              # Project documentation
```

---

## Usage Guide

### 1. **Overview**

- Navigate to the "Overview" tab to explore key insights into the data.
- Use filters to refine the visualizations dynamically.

### 2. **Regression**

- Navigate to the "Regression" tab.
- Adjust the sliders in the sidebar to input feature values.
- Click "Predict Rating" to see the predicted attraction rating.

### 3. **Classification**

- Navigate to the "Classification" tab.
- Adjust the sliders to input features.
- Click "Predict Visit Mode" to get the predicted mode of visit.
- View the detailed classification report for performance metrics.

### 4. **Recommendations**

- Navigate to the "Recommendations" tab.
- Enter a valid User ID in the sidebar.
- Click "Get Recommendations" to view the top 5 recommended attractions with estimated ratings.

---

## Data Sources

- **Merged\_TourismExperience.csv**: Contains cleaned and preprocessed data, including user demographics, attraction details, visit modes, and ratings.

---

## Key Technologies

- **Streamlit**: For building the interactive dashboard.
- **Pandas**: For data manipulation and cleaning.
- **Seaborn & Matplotlib**: For data visualizations.
- **Scikit-learn**: For machine learning models (Regression and Classification).
- **Surprise**: For building the collaborative filtering recommendation system.

---

## Future Enhancements

- Add advanced recommendation models, such as hybrid systems combining collaborative and content-based filtering.
- Integrate geographic visualizations for better location-based insights.
- Provide additional filtering options for personalized analytics.

---

## Author

Senthilkumar Sundaram



