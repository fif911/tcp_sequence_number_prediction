import numpy as np
import plotly.graph_objects as go
from pykalman import KalmanFilter
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def create_input_vector(seq_numbers, n):
    return np.array([seq_numbers[n - 2], seq_numbers[n - 1], seq_numbers[n]])


if __name__ == '__main__':
    # read CSV file
    data = np.genfromtxt('dump.logs', delimiter=',', dtype=None, encoding=None)
    seq_numbers = data[:, 1]
    seq_numbers = np.array([int(seq.replace('seq ', '')) for seq in seq_numbers])

    # Initialize vectors with None values
    x = [None] * len(seq_numbers)
    y = [None] * len(seq_numbers)
    z = [None] * len(seq_numbers)

    for n in range(3, len(seq_numbers)):
        x[n] = seq_numbers[n - 2] - seq_numbers[n - 3]
        y[n] = seq_numbers[n - 1] - seq_numbers[n - 2]
        z[n] = seq_numbers[n] - seq_numbers[n - 1]

    # cut first 3 values
    x = x[3:]
    y = y[3:]
    z = z[3:]
    seq_numbers = seq_numbers[3:]

    # Create an array of indices
    n = np.arange(len(seq_numbers))

    # Reshape the arrays for input into the PolynomialFeatures transformer
    features = np.array([n, x, y, z]).T

    # Reshape the arrays for input into the Linear Regression model
    seq_numbers = seq_numbers.reshape(-1, 1)

    # Split the data into training and testing sets
    features_train, features_test, seq_numbers_train, seq_numbers_test = train_test_split(
        features, seq_numbers, test_size=0.2, random_state=0
    )

    degrees = [3]
    for degree in degrees:
        # Create a pipeline with Polynomial Features, Standard Scaler, and Ridge Regression
        model = make_pipeline(PolynomialFeatures(degree), StandardScaler(), Ridge(alpha=1.0))

        # Fit the model
        model.fit(features_train, seq_numbers_train)

        # Print coefficients
        print(f'Degree {degree} Polynomial Model Coefficients:')
        for i, coef in enumerate(model.named_steps['ridge'].coef_[0]):
            print(f'Coefficient {i}: {coef}')

        # Plot the expected and predicted lines
        fig = go.Figure()

        # Plot the expected line
        fig.add_trace(go.Scatter(x=n, y=seq_numbers.flatten(), mode='markers', name='Actual'))

        # Plot the predicted line
        seq_numbers_pred = model.predict(features)
        fig.add_trace(go.Scatter(x=n, y=seq_numbers_pred.flatten(), mode='markers', name='Predicted'))

        fig.update_layout(title=f'Degree {degree} Polynomial Regression',
                          xaxis_title='Index',
                          yaxis_title='Sequence Number')

        fig.show()

        # Evaluate the model
        mse = mean_squared_error(seq_numbers_test, model.predict(features_test))
        r2 = r2_score(seq_numbers_test, model.predict(features_test))

        print(f"Degree {degree} - Mean Squared Error: {mse}, R-squared: {r2}")

        predicted_successfully = 0
        predictions_total = 0
        for i in range(10, len(seq_numbers) - 1):
            # Reshape the input for prediction
            feature_pred = np.array([[i, x[i], y[i], z[i]]])

            # Use the polynomial regression model for prediction
            predicted = int(model.predict(feature_pred)[0][0])
            if predicted == seq_numbers[i + 1]:
                predicted_successfully += 1
            print("Predicted: ", predicted, "Actual: ", seq_numbers[i + 1])
            predictions_total += 1

        print(f'Degree {degree} - Predicted successfully: {predicted_successfully}')
        print(f'Degree {degree} - Predictions total: {predictions_total}')
        print(f'Degree {degree} - Accuracy: {predicted_successfully / predictions_total}')
        print('\n')
