import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import psycopg2
from scipy.stats import spearmanr


host = "localhost"
port = "5432"
database = "Crop_Prediction"
username = "postgres"
password = "password"

engine = create_engine(f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}')



choose = st.sidebar.selectbox("", ['Predicting Production', 'Exploratory Data Analysis (EDA)'])

if choose == 'Exploratory Data Analysis (EDA)':
    st.title("Crop Production Data Analysis")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        [
            "Crop Types",
            "Geographical Distribution",
            "Yearly Trends",
            "Growth Analysis",
            "Environmental Relationships",
            'Input-Output Relationships',
            'Comparative Analysis',
            'Productivity Analysis',
            'productivity ratios',
            'Outliers and Anomalies'
        ]
    )

    if analysis_type == 'Crop Types':
        query = """
        SELECT 
            item,
            SUM(value) AS total_production
        FROM crop_production
        WHERE element = 'Production'
        GROUP BY item
        ORDER BY total_production DESC
        """

        crop_distribution_df = pd.read_sql_query(query, engine)

        top10_crops = crop_distribution_df.head(10)

        bottom10_crops = crop_distribution_df.tail(10)

        check = st.radio('',['Most Cultivated Crops', 'Least Cultivated Crops'])
        if check == 'Most Cultivated Crops':
            fig1 = plt.figure(figsize=(10, 6))
            sns.barplot(y='item', x='total_production', data=top10_crops, palette='Greens', hue='total_production',
                        legend=False)
            plt.title('Top 10 Most Cultivated Crops (by Total Production)')
            plt.xlabel('Total Production (tons)')
            plt.ylabel('Crop')
            plt.tight_layout()
            st.pyplot(fig1)

        else:
            fig2 = plt.figure(figsize=(10, 6))
            sns.barplot(y='item', x='total_production', data=bottom10_crops, palette='Reds', hue='total_production',
                        legend=False)
            plt.title('Top 10 Least Cultivated Crops (by Total Production)')
            plt.xlabel('Total Production (tons)')
            plt.ylabel('Crop')
            plt.tight_layout()
            st.pyplot(fig2)

    elif analysis_type == 'Geographical Distribution':
        query = """
        SELECT 
            area,
            SUM(value) AS total_production
        FROM crop_production
        WHERE element = 'Production'
        GROUP BY area
        ORDER BY total_production DESC
        """

        production_by_country = pd.read_sql_query(query, engine)

        top15_countries = production_by_country.head(15)

        fig = plt.figure(figsize=(12, 6))
        sns.barplot(y='area', x='total_production', data=top15_countries, palette='coolwarm', hue='total_production',
                    legend=False)
        plt.title('Top 15 Countries by Total Crop Production')
        plt.xlabel('Total Production (tons)')
        plt.ylabel('Country')
        plt.tight_layout()
        st.pyplot(fig)

    elif analysis_type == 'Yearly Trends':
        query = """
        SELECT 
            year,
            element,
            CASE 
                WHEN element = 'Yield' THEN AVG(value)
                ELSE SUM(value)
            END AS total_value
        FROM crop_production
        WHERE element IN ('Area harvested', 'Yield', 'Production')
        GROUP BY year, element
        ORDER BY year, element
        """

        yearly_trends_df = pd.read_sql_query(query, engine)

        fig = plt.figure(figsize=(12, 6))
        sns.lineplot(data=yearly_trends_df, x='year', y='total_value', hue='element', marker='o')
        plt.title('Yearly Trends in Area Harvested, Yield, and Production')
        plt.xlabel('Year')
        plt.ylabel('Total Value')
        plt.legend(title='Element')
        plt.xticks(sorted(yearly_trends_df['year'].unique()))

        plt.tight_layout()
        st.pyplot(fig)

    elif analysis_type == 'Growth Analysis':
        query = """
        SELECT 
            item,
            year,
            SUM(value) AS total_production
        FROM crop_production
        WHERE element = 'Production'
        GROUP BY item, year
        HAVING COUNT(*) > 1
        ORDER BY item, year
        """

        crop_trends_df = pd.read_sql_query(query, engine)

        top_crops = (
            crop_trends_df.groupby('item')['total_production']
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )

        filtered_df = crop_trends_df[crop_trends_df['item'].isin(top_crops)]

        fig = plt.figure(figsize=(12, 6))
        sns.lineplot(data=filtered_df, x='year', y='total_production', hue='item', marker='o')
        plt.title('Production Trends of Top 5 Crops Over the Years')
        plt.xlabel('Year')
        plt.ylabel('Total Production (tons)')
        plt.legend(title='Crop Item')

        plt.xticks(sorted(filtered_df['year'].unique()))

        plt.tight_layout()
        st.pyplot(fig)

    elif analysis_type == 'Input-Output Relationships':
        area_query = """
        SELECT 
            item,
            year,
            SUM(value) AS total_area
        FROM crop_production
        WHERE element = 'Area harvested'
        GROUP BY item, year
        """
        area_df = pd.read_sql_query(area_query, engine)

        yield_query = """
        SELECT 
            item,
            year,
            AVG(value) AS avg_yield
        FROM crop_production
        WHERE element = 'Yield'
        GROUP BY item, year
        """
        yield_df = pd.read_sql_query(yield_query, engine)

        production_query = """
        SELECT 
            item,
            year,
            SUM(value) AS total_production
        FROM crop_production
        WHERE element = 'Production'
        GROUP BY item, year
        """
        production_df = pd.read_sql_query(production_query, engine)

        merged_df = area_df.merge(yield_df, on=['item', 'year']) \
            .merge(production_df, on=['item', 'year'])

        correlation_matrix = merged_df[['total_area', 'avg_yield', 'total_production']].corr()

        fig = plt.figure(figsize=(8, 5))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Between Area Harvested, Yield, and Production')
        plt.tight_layout()
        st.pyplot(fig)

    elif analysis_type == 'Environmental Relationships':
        query = """
        SELECT
            area,
            item,
            year,
            SUM(CASE WHEN element = 'Area harvested' THEN value ELSE 0 END) AS area_harvested,
            SUM(CASE WHEN element = 'Yield' THEN value ELSE 0 END) AS yield
        FROM crop_production
        WHERE element IN ('Area harvested', 'Yield')
        GROUP BY area, item, year
        HAVING
            SUM(CASE WHEN element = 'Area harvested' THEN value ELSE 0 END) > 0 AND
            SUM(CASE WHEN element = 'Yield' THEN value ELSE 0 END) > 0
        ORDER BY year;
        """

        df = pd.read_sql_query(query, engine)

        # Drop rows with 0 or null values
        df = df[(df['area_harvested'] > 0) & (df['yield'] > 0)].copy()

        # Correlation
        pearson_corr = df['area_harvested'].corr(df['yield'], method='pearson')
        spearman_corr = df['area_harvested'].corr(df['yield'], method='spearman')
        print(f"Pearson Correlation: {pearson_corr:.2f}")
        print(f"Spearman Correlation: {spearman_corr:.2f}")

        # Regression Plot
        fig = plt.figure(figsize=(10, 6))
        sns.regplot(x='area_harvested', y='yield', data=df, scatter_kws={'alpha': 0.6})
        plt.title('Relationship Between Area Harvested and Yield')
        plt.xlabel('Area Harvested (ha)')
        plt.ylabel('Yield (hg/ha)')
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(fig)

    elif analysis_type == 'Comparative Analysis':

        check = st.selectbox('',['Across Crops', 'Across Regions'])

        if check == 'Across Crops':
            query = """
            SELECT 
                item,
                AVG(value) AS avg_yield
            FROM crop_production
            WHERE element = 'Yield'
            GROUP BY item
            ORDER BY avg_yield DESC
            """

            yield_df = pd.read_sql_query(query, engine)

            top10_yield = yield_df.head(10)

            bottom10_yield = yield_df.tail(10)

            high_low = st.radio('', ['High-Yield Crops', 'Low-Yield Crops'])

            if high_low == 'High-Yield Crops':
                fig1 = plt.figure(figsize=(10, 6))
                sns.barplot(x='avg_yield', y='item', data=top10_yield, palette='mako', hue='avg_yield', legend=False)
                plt.title('Top 10 High-Yield Crops (kg/ha)')
                plt.xlabel('Average Yield (kg/ha)')
                plt.ylabel('Crop')
                plt.tight_layout()
                st.pyplot(fig1)
            else:
                fig2 = plt.figure(figsize=(10, 6))
                sns.barplot(x='avg_yield', y='item', data=bottom10_yield, palette="crest", hue='avg_yield', legend=False)
                plt.title('Bottom 10 Low-Yield Crops (kg/ha)')
                plt.xlabel('Average Yield (kg/ha)')
                plt.ylabel('Crop')
                plt.tight_layout()
                st.pyplot(fig2)
        else:
            query = """
            SELECT 
                area,
                SUM(value) AS total_production
            FROM crop_production
            WHERE element = 'Production'
            GROUP BY area
            ORDER BY total_production DESC
            """

            region_production_df = pd.read_sql_query(query, engine)

            top15_regions = region_production_df.head(15)

            fig = plt.figure(figsize=(12, 6))
            sns.barplot(x='total_production', y='area', data=top15_regions, palette='crest', hue='total_production')
            plt.title('Top 15 Most Productive Countries (Total Crop Production)')
            plt.xlabel('Total Production (tons)')
            plt.ylabel('Country / Region')
            plt.tight_layout()
            st.pyplot(fig)


    elif analysis_type == 'Productivity Analysis':

        high_low = st.radio('', ['High-Yield Crops', 'High-Yield Regions'])

        if high_low == 'High-Yield Crops':
            query_crop = """
            SELECT 
                item,
                AVG(value) AS avg_yield
            FROM crop_production
            WHERE element = 'Yield'
            GROUP BY item
            ORDER BY avg_yield DESC
            """

            yield_by_crop = pd.read_sql_query(query_crop, engine)

            top10_crops = yield_by_crop.head(10)

            fig = plt.figure(figsize=(10, 6))
            sns.barplot(x='avg_yield', y='item', data=top10_crops, palette='Spectral', hue='avg_yield', legend=False)
            plt.title('Top 10 High-Yield Crops')
            plt.xlabel('Average Yield (kg/ha)')
            plt.ylabel('Crop')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            query_region = """
            SELECT 
                area,
                AVG(value) AS avg_yield
            FROM crop_production
            WHERE element = 'Yield'
            GROUP BY area
            ORDER BY avg_yield DESC
            """

            yield_by_region = pd.read_sql_query(query_region, engine)

            top10_regions = yield_by_region.head(10)

            fig = plt.figure(figsize=(10, 6))
            sns.barplot(x='avg_yield', y='area', data=top10_regions, palette='Spectral', hue='avg_yield', legend=False)
            plt.title('Top 10 High-Yield Regions')
            plt.xlabel('Average Yield (kg/ha)')
            plt.ylabel('Region')
            plt.tight_layout()
            st.pyplot(fig)

    elif analysis_type == 'productivity ratios':
        production_query = """
        SELECT
            item,
            year,
            SUM(value) AS total_production
        FROM crop_production
        WHERE element = 'Production'
        GROUP BY item, year
        """

        production_df = pd.read_sql_query(production_query, engine)

        area_query = """
        SELECT
            item,
            year,
            SUM(value) AS total_area
        FROM crop_production
        WHERE element = 'Area harvested'
        GROUP BY item, year
        """

        area_df = pd.read_sql_query(area_query, engine)

        merged_df = pd.merge(production_df, area_df, on=['item', 'year'])

        merged_df['productivity_ratio'] = merged_df['total_production'] / merged_df['total_area']

        yield_query = """
        SELECT
            item,
            year,
            AVG(value) AS avg_yield
        FROM crop_production
        WHERE element = 'Yield'
        GROUP BY item, year
        """

        yield_df = pd.read_sql_query(yield_query, engine)

        final_df = pd.merge(merged_df, yield_df, on=['item', 'year'])

        corr = final_df[['productivity_ratio', 'avg_yield']].corr().iloc[0, 1]
        print(f"Correlation between productivity ratio and average yield: {corr:.4f}")

        sample_crops = final_df['item'].value_counts().head(5).index.tolist()
        plot_df = final_df[final_df['item'].isin(sample_crops)]

        fig = plt.figure(figsize=(12, 6))

        sns.lineplot(data=plot_df, x='year', y='productivity_ratio', hue='item', marker='o', linestyle='--')

        sns.lineplot(data=plot_df, x='year', y='avg_yield', hue='item', marker='o', legend=False)

        plt.title('Productivity Ratio vs Average Yield Over Years (Sample Crops)')
        plt.xlabel('Year')
        plt.ylabel('Value (Production/Area or Yield in kg/ha)')
        plt.legend(title='Crop')
        plt.xticks(sorted(final_df['year'].unique()))

        plt.tight_layout()
        st.pyplot(fig)

    else:
        query = """
        SELECT 
            area,
            item,
            year,
            SUM(CASE WHEN element = 'Production' THEN value ELSE 0 END) AS production,
            SUM(CASE WHEN element = 'Yield' THEN value ELSE 0 END) AS yield
        FROM crop_production
        WHERE element IN ('Production', 'Yield')
        GROUP BY area, item, year
        HAVING 
            SUM(CASE WHEN element = 'Production' THEN value ELSE 0 END) > 0 AND
            SUM(CASE WHEN element = 'Yield' THEN value ELSE 0 END) > 0
        ORDER BY year;
    
        """

        from scipy.stats import zscore

        df = pd.read_sql_query(query, engine)
        df = df[(df['production'] > 0) & (df['yield'] > 0)].copy()

        # Z-scores
        df['production_z'] = zscore(df['production'])
        df['yield_z'] = zscore(df['yield'])

        # Flagging anomalies (Z-score > 3 or < -3)
        anomalies = df[(df['production_z'].abs() > 3) | (df['yield_z'].abs() > 3)]

        print(f"Total Anomalies Detected: {len(anomalies)}")

        # Plot with anomalies
        fig = plt.figure(figsize=(12, 6))

        # Plot normal points (colored by item)
        sns.scatterplot(data=df, x='year', y='yield', hue='item', alpha=0.5, legend=False)

        # Plot anomalies as red Xs
        sns.scatterplot(
            data=anomalies,
            x='year',
            y='yield',
            color='red',
            marker='X',
            s=100,
            label='Anomalies',  # Only one manual label
            legend=False
        )

        plt.title('Yield Over Years with Anomaly Detection')
        plt.xlabel('Year')
        plt.ylabel('Yield')
        plt.xticks(sorted(df['year'].unique()))
        plt.tight_layout()
        st.pyplot(fig)

else:
    engine = create_engine(f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}')

    crop_query = """
    SELECT * FROM crop_production
    """

    crop_df = pd.read_sql_query(crop_query, engine)

    pivot_df = crop_df.pivot_table(
        index=['area', 'item', 'year'],
        columns='element',
        values='value',
        aggfunc='first'
    ).reset_index()

    pivot_df.rename(columns={
        'area': 'Area',
        'item': 'Item',
        'year': 'Year',
        'Area harvested': 'Area_Harvested',
    }, inplace=True)
    pivot_df = pivot_df[['Area', 'Item', 'Year', 'Area_Harvested', 'Yield', 'Production']].copy()

    pivot_df_clean = pivot_df.dropna(subset=['Area_Harvested', 'Yield', 'Production'])

    X = pivot_df_clean.drop(['Production', 'Yield'], axis=1)
    y = pivot_df_clean['Production']

    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.tree import DecisionTreeRegressor


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    ct = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [0, 1])
        ],
        remainder='passthrough'
    )

    X_train_encoded = ct.fit_transform(X_train)
    X_test_encoded = ct.transform(X_test)


    decisiontree_model = DecisionTreeRegressor(random_state=0)
    decisiontree_model.fit(X_train_encoded, y_train)



    st.title("Crop Yield Prediction")

    st.subheader("Enter Input Details")
    area = st.selectbox('Select Area', sorted(pivot_df_clean['Area'].unique()))
    item = st.selectbox('Select Item', sorted(pivot_df_clean['Item'].unique()))
    year = st.number_input('Year', min_value=2019, max_value=2023, step=1)
    area_harvested = st.number_input('Area Harvested', min_value=0.0, step=100.0)


    if st.button('Predict Production'):
        input_df = pd.DataFrame([{
            'Area': area,
            'Item': item,
            'Year': year,
            'Area_Harvested': area_harvested
        }])

        transformed_input = ct.transform(input_df)
        prediction = decisiontree_model.predict(transformed_input)

        st.success(f"Predicted Production (in tons): {prediction[0]:,.2f}")