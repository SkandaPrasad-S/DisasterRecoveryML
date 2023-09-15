# Disaster Response Pipeline Project Documentation

**Project Overview**

This project aims to create an end-to-end data pipeline for classifying disaster-related messages into various categories. It includes three main components: an ETL (Extract, Transform, Load) pipeline, an ML (Machine Learning) pipeline, and a Flask web application for user interaction.

**Project Structure**

The project is structured into three main parts:

1. **ETL Pipeline (process_data.py)**
    - The ETL pipeline is responsible for loading and preprocessing the input data.
    - It consists of the following steps:
        - Loading the messages and categories datasets.
        - Merging the two datasets based on the common 'id' column.
        - Cleaning the data by splitting categories, converting them to binary values, and removing duplicates.
        - Storing the cleaned data in an SQLite database.

2. **ML Pipeline (train_classifier.py)**
    - The ML pipeline builds and trains a classification model on the preprocessed data.
    - It consists of the following steps:
        - Loading data from the SQLite database.
        - Splitting the dataset into training and testing sets.
        - Building a text processing and machine learning pipeline, which includes tokenization, TF-IDF transformation, and a multi-output classifier (Random Forest in this case).
        - Using GridSearchCV for hyperparameter tuning.
        - Training and evaluating the model.
        - Saving the final model as a pickle file.

3. **Flask Web App (app.py)**
    - The Flask web application provides a user-friendly interface for interacting with the trained model.
    - It consists of the following parts:
        - Loading the saved model and the preprocessed data.
        - Creating data visualizations (e.g., genre distribution) using Plotly.
        - Handling user queries and displaying model predictions.
        - Running the web app on a local server.

***

**Project Execution**

To execute the project successfully, follow these steps:

1. Run the ETL pipeline script (process_data.py) to preprocess the data and store it in an SQLite database.

    ```shell
    python process_data.py messages.csv categories.csv DisasterResponse.db
    ```

2. Run the ML pipeline script (train_classifier.py) to train and save the model.

    ```shell
    python train_classifier.py DisasterResponse.db classifier.pkl
    ```

3. Start the Flask web app by running app.py.

    ```shell
    python app.py
    ```

4. Access the web app through a web browser (usually at http://localhost:3000).

***

**Project Achievements**

This project demonstrates several key skills and accomplishments:

- **Data Cleaning and Preprocessing:** The ETL pipeline effectively loads, merges, and cleans the dataset, making it ready for model training.

- **Machine Learning Pipeline:** The ML pipeline constructs a robust text processing and classification model using a multi-output classifier. It also utilizes hyperparameter tuning to optimize model performance.

- **Interactive Web Application:** The Flask web app provides a user-friendly interface for classifying messages and includes data visualizations for insights.

- **Persistence and Reusability:** The project successfully saves and loads both the trained model and preprocessed data, making it easy to deploy and reuse.

- **Documentation:** This comprehensive documentation highlights the project's objectives, structure, and execution steps.

Overall, this project is a great achievement as it combines data engineering, machine learning, and web development skills to create a functional and user-friendly tool for disaster response classification. It showcases the ability to handle real-world data, build effective machine learning models, and provide practical solutions for a critical domain.