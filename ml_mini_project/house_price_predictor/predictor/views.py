from django.shortcuts import render
# predictor/views.py
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import mean_absolute_error
import time

# Global variable to store the dataFrame
df = None

def upload_dataset(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        myfile = request.FILES['csv_file']
        algorithm = request.POST.get('algorithm')
        
        try:
            # Read the CSV into pandas DataFrame
            df = pd.read_csv(myfile)
            
            # Show DataFrame summary statistics
            summary = df.describe().to_html(classes='table table-striped')
            # Selecting features and target
            X = df[['area', 'bedrooms', 'bathrooms', 'floors', 'age']]  # Features
            y = df['price']  # Target variable

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Handling selected algorithms
            if algorithm == 'linear_regression':
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                model_type = 'Linear Regression'
                
                start_time = time.time()
                model.fit(X_train, y_train)   
                training_time = time.time() - start_time
               # Predict and calculate prediction time
                start_time = time.time()
                y_pred = model.predict(X_test)
                prediction_time = time.time() - start_time
                mse = mean_squared_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Calculate training and testing errors
                train_pred = model.predict(X_train)
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, y_pred)

                # Add noise to evaluate robustness
                noise = np.random.normal(0, 0.1, X_test.shape)
                noisy_X_test = X_test + noise
                noisy_y_pred = model.predict(noisy_X_test)
                noisy_mse = mean_squared_error(y_test, noisy_y_pred)
                evaluation = f"""Mean Squared Error (MSE): {mse:.2f}
                                 R-squared Score: {r2:.2f}
                                 Mean Absolute Error (MAE): {mae:.2f}
                                 Training Time: {training_time:.2f} seconds
                                 Prediction Time: {prediction_time:.2f} seconds
                                 Training MSE: {train_mse:.2f}
                                 Testing MSE: {test_mse:.2f}
                                 Noisy Data MSE: {noisy_mse:.2f}"""

            elif algorithm == 'logistic_regression':
                start_time = time.time()
                model = LogisticRegression(max_iter=200)
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                start_time = time.time()
                y_pred = model.predict(X_test)
                prediction_time = time.time() - start_time
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                noise = np.random.normal(0, 0.1, X_test.shape)
                noisy_X_test = X_test + noise
                noisy_y_pred = model.predict(noisy_X_test)
                noisy_accuracy = accuracy_score(y_test, noisy_y_pred)
                
                model_type = 'Logistic Regression'
                evaluation = f"""Accuracy: {accuracy:.2f}
                                 Precision: {precision:.2f}
                                 Recall: {recall:.2f}
                                 F1 Score: {f1:.2f}
                                 Training Time: {training_time:.2f} seconds
                                 Prediction Time: {prediction_time:.2f} seconds
                                 Robustness to Noise (Accuracy with Noisy Data): {noisy_accuracy:.2f}
                                 Confusion Matrix: {conf_matrix.tolist()}"""

            elif algorithm == 'polynomial_regression':
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X_train)
                model = LinearRegression()
                
                start_time = time.time()
                model.fit(X_poly, y_train)
                training_time = time.time() - start_time

                start_time = time.time()
                y_pred = model.predict(poly.transform(X_test))
                prediction_time = time.time() - start_time

                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                train_pred = model.predict(X_poly)
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, y_pred)

                noise = np.random.normal(0, 0.1, X_test.shape)
                noisy_X_test = poly.transform(X_test + noise)
                noisy_y_pred = model.predict(noisy_X_test)
                noisy_mse = mean_squared_error(y_test, noisy_y_pred)

                model_type = 'Polynomial Regression'
                evaluation = f"""Mean Squared Error (MSE): {mse:.2f}
                                 R-squared Score: {r2:.2f}
                                 Mean Absolute Error (MAE): {mae:.2f}
                                 Training Time: {training_time:.2f} seconds
                                 Prediction Time: {prediction_time:.2f} seconds
                                 Training MSE: {train_mse:.2f}
                                 Testing MSE: {test_mse:.2f}
                                 Noisy Data MSE: {noisy_mse:.2f}"""

            elif algorithm == 'decision_tree':
                model = DecisionTreeRegressor(random_state=42)
                
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time

                start_time = time.time()
                y_pred = model.predict(X_test)
                prediction_time = time.time() - start_time

                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                train_pred = model.predict(X_train)
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, y_pred)

                noise = np.random.normal(0, 0.1, X_test.shape)
                noisy_X_test = X_test + noise
                noisy_y_pred = model.predict(noisy_X_test)
                noisy_mse = mean_squared_error(y_test, noisy_y_pred)

                model_type = 'Decision Tree'
                evaluation = f"""Mean Squared Error (MSE): {mse:.2f}
                                 R-squared Score: {r2:.2f}
                                 Mean Absolute Error (MAE): {mae:.2f}
                                 Training Time: {training_time:.2f} seconds
                                 Prediction Time: {prediction_time:.2f} seconds
                                 Training MSE: {train_mse:.2f}
                                 Testing MSE: {test_mse:.2f}
                                 Noisy Data MSE: {noisy_mse:.2f}"""

            elif algorithm == 'random_forest':
                model = RandomForestRegressor(random_state=42)
                
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time

                start_time = time.time()
                y_pred = model.predict(X_test)
                prediction_time = time.time() - start_time

                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                train_pred = model.predict(X_train)
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, y_pred)

                noise = np.random.normal(0, 0.1, X_test.shape)
                noisy_X_test = X_test + noise
                noisy_y_pred = model.predict(noisy_X_test)
                noisy_mse = mean_squared_error(y_test, noisy_y_pred)

                model_type = 'Random Forest'
                evaluation = f"""Mean Squared Error (MSE): {mse:.2f}
                                 R-squared Score: {r2:.2f}
                                 Mean Absolute Error (MAE): {mae:.2f}
                                 Training Time: {training_time:.2f} seconds
                                 Prediction Time: {prediction_time:.2f} seconds
                                 Training MSE: {train_mse:.2f}
                                 Testing MSE: {test_mse:.2f}
                                 Noisy Data MSE: {noisy_mse:.2f}"""

            elif algorithm == 'knn':
                model = KNeighborsRegressor()
                
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time

                start_time = time.time()
                y_pred = model.predict(X_test)
                prediction_time = time.time() - start_time

                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                train_pred = model.predict(X_train)
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, y_pred)

                noise = np.random.normal(0, 0.1, X_test.shape)
                noisy_X_test = X_test + noise
                noisy_y_pred = model.predict(noisy_X_test)
                noisy_mse = mean_squared_error(y_test, noisy_y_pred)

                model_type = 'K-Nearest Neighbors'
                evaluation = f"""Mean Squared Error (MSE): {mse:.2f}
                                 R-squared Score: {r2:.2f}
                                 Mean Absolute Error (MAE): {mae:.2f}
                                 Training Time: {training_time:.2f} seconds
                                 Prediction Time: {prediction_time:.2f} seconds
                                 Training MSE: {train_mse:.2f}
                                 Testing MSE: {test_mse:.2f}
                                 Noisy Data MSE: {noisy_mse:.2f}"""

            elif algorithm == 'naive_bayes':
                model = GaussianNB()
                
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time

                start_time = time.time()
                y_pred = model.predict(X_test)
                prediction_time = time.time() - start_time

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
                conf_matrix = confusion_matrix(y_test, y_pred)

                noise = np.random.normal(0, 0.1, X_test.shape)
                noisy_X_test = X_test + noise
                noisy_y_pred = model.predict(noisy_X_test)
                noisy_accuracy = accuracy_score(y_test, noisy_y_pred)

                model_type = 'Naive Bayes'
                evaluation = f"""Accuracy: {accuracy:.2f}
                                 Precision: {precision:.2f}
                                 Recall: {recall:.2f}
                                 F1 Score: {f1:.2f}
                                 Training Time: {training_time:.2f} seconds
                                 Prediction Time: {prediction_time:.2f} seconds
                                 Robustness to Noise (Accuracy with Noisy Data): {noisy_accuracy:.2f}
                                 Confusion Matrix: {conf_matrix.tolist()}"""

            else:
                model_type = 'No model selected'
                evaluation = 'Please select an algorithm to run.'

            # Return response with results
            return render(request, 'upload_dataset.html', {
                'summary': summary,
                'model_type': model_type,
                'evaluation': evaluation,
                'csv_file': myfile
            })
        
        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            return render(request, 'upload_dataset.html', {'error': error_msg})
    
    return render(request, 'upload_dataset.html')
