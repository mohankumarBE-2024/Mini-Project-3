# Mini-Project-3
Predicting Crop Production Based  on Agricultural Data 

📌 Overview
This project aims to predict agricultural crop production using historical data from FAOSTAT.
The workflow includes preprocessing the raw .csv data, storing it in a PostgreSQL database, training machine learning models, and deploying an interactive prediction dashboard using Streamlit.
The goal is to assist farmers, policymakers, and researchers with accurate crop yield predictions based on factors like area, crop type, element, and year.

🛠️ Technologies Used
Python (Pandas, NumPy): For data manipulation and preprocessing.

scikit-learn: For building and evaluating machine learning models (Linear Regression, Random Forest, etc.).

PostgreSQL: For structured storage of cleaned agricultural data.

psycopg2: For PostgreSQL database connection and execution.

SQLAlchemy: For seamless integration of PostgreSQL with Pandas.

Streamlit: For building a user-friendly web app for predictions.

Matplotlib & Seaborn: For visual exploration and insights into the dataset.

🔍 Steps Involved
1. Data Preprocessing
Loaded the FAOSTAT .csv dataset using Pandas.

Removed irrelevant columns like "Note" and dropped fully empty rows.

Selected relevant columns such as Domain, Area, Element, Item, Year, Unit, Value, and Flag.

Filled missing Unit values by matching based on Area, Item, and Element.

Imputed missing Value using group-wise mean based on Item, Year, Element, and Unit.

Replaced values with flag 'M' (missing) as 0 for modeling consistency.

2. PostgreSQL Table Creation & Data Insertion
Created a table named crop_production using appropriate data types (TEXT, INTEGER, FLOAT).

Used psycopg2 to execute the SQL table creation script.

Inserted the cleaned DataFrame into the PostgreSQL database using SQLAlchemy engine and to_sql().

3. Machine Learning Model Training
Extracted features like Area, Item, Element, and Year.

Applied one-hot encoding to categorical features.

Trained models like:

Linear Regression

Random Forest Regressor

Evaluated performance using metrics like RMSE and R² Score.

4. Streamlit Dashboard Development
Built an interactive web app using Streamlit.

Included dropdowns for selecting Area, Item, Element, and Year dynamically from unique values.

On prediction, the app:

Processes inputs into model-ready format

Applies the trained model

Displays predicted production value with proper formatting

Designed the UI with user-friendly layouts, success messages, and clear headings.

