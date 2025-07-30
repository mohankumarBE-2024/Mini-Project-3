# Mini-Project-3: Predicting Crop Production Based on Agricultural Data

## Overview
This project focuses on predicting crop production (in tons) for different regions and years based on historical agricultural data. The data includes features like Area, Item (crop type), Area Harvested, and Year. The primary goal is to provide a tool that helps estimate future crop production using machine learning models and interactive dashboards.

The dataset is sourced from raw .csv data and processed using Pandas before being stored in a PostgreSQL database. An interactive prediction interface and exploratory data analysis (EDA) views are developed using Streamlit. The project aims to deliver both insightful analytics and predictive modeling within a user-friendly web app.

## Technologies Used
- **Python** (Pandas, NumPy): For data manipulation and preprocessing.
- **scikit-learn**: For building and evaluating machine learning models (Linear Regression, Random Forest, etc.).
- **PostgreSQL**: For structured storage of cleaned agricultural data.
- **psycopg2**: For PostgreSQL database connection and execution.
- **SQLAlchemy**: For seamless integration of PostgreSQL with Pandas.
- **Streamlit**: For building a user-friendly web app for predictions.
- **Matplotlib & Seaborn**: For visual exploration and insights into the dataset.

## Steps Involved

### 1. Data Preprocessing
- Loaded the FAOSTAT .csv dataset using Pandas.
- Removed irrelevant columns like "Note" and dropped fully empty rows.
- Selected relevant columns such as Domain, Area, Element, Item, Year, Unit, Value, and Flag.
- Filled missing Unit values by matching based on Area, Item, and Element.
- Imputed missing Value using group-wise mean based on Item, Year, Element, and Unit.
- Replaced values with flag 'M' (missing) as 0 for modeling consistency.

### 2. PostgreSQL Table Creation & Data Insertion
- Created a table named `crop_production` using appropriate data types (TEXT, INTEGER, FLOAT).
- Used psycopg2 to execute the SQL table creation script.
- Inserted the cleaned DataFrame into the PostgreSQL database using SQLAlchemy engine and `to_sql()`.

### 3. Machine Learning Model Training
- Extracted features like Area, Item, Element, and Year.
- Applied one-hot encoding to categorical features.
- Trained models including:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- Evaluated performance using metrics like R² Score.

### 4. Streamlit Dashboard Development
- Built an interactive web app using Streamlit.
- Included dropdowns for selecting Area, Item, Element, and Year dynamically from unique values.
- On prediction, the app:
  - Processes inputs into model-ready format
  - Applies the trained model
  - Displays predicted production value with proper formatting
- Designed the UI with user-friendly layouts, success messages, and clear headings.
