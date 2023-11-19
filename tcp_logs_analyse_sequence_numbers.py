import ctypes
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots

PLOT_OR_PREDICT = 'predict'  # 'plot' or 'predict'


def predict_next_sn_linear(last_sn):
    # Calculate 3rd point based on 2 points that are linearly dependent
    print("last_sn: ", last_sn)
    x1, y1 = 1, last_sn[0]  # point 1
    x2, y2 = 2, last_sn[1]  # point 2
    m = (y2 - y1) / (x2 - x1)  # slope

    # Use the slope to predict the next point
    x3 = 3  # x-coordinate of the next point
    y3 = y2 + m * (x3 - x2)  # linear equation: y = mx + c, where c = y2 - m * x2

    return y3


def plot_sequence_numbers(sn_list: List[int], predicted_sn_list: List[bool]):
    # Create a trace for actual sequence numbers
    trace_actual = go.Scatter(
        x=list(range(len(sn_list))),
        y=sn_list,
        mode='markers',
        name='Actual Sequence Numbers',
        line=dict(color='blue')
    )

    # Create a trace for predicted sequence numbers
    # trace_predicted = go.Scatter(
    #     x=list(range(len(sn_list))),
    #     y=[predicted_sn_list[i] * sn_list[i] for i in range(len(sn_list))],
    #     mode='lines',
    #     name='Predicted Sequence Numbers',
    #     line=dict(color='green')
    # )

    # Create a trace for incorrectly predicted sequence numbers
    trace_incorrect = go.Scatter(
        x=[i for i in range(len(sn_list)) if not predicted_sn_list[i]],
        y=[sn_list[i] for i in range(len(sn_list)) if not predicted_sn_list[i]],
        mode='markers',
        name='Incorrect Predictions',
        marker=dict(color='red', size=8)
    )

    # Layout settings
    layout = go.Layout(
        title='Actual vs Predicted Sequence Numbers',
        xaxis=dict(title='Batch Index'),
        yaxis=dict(title='Sequence Number'),
    )

    # Combine the traces into a data list
    data = [trace_actual, trace_incorrect]

    # Create the figure
    fig = go.Figure(data=data, layout=layout)

    # Show the figure
    fig.show()


# Function to predict the next value in the sequence
def predict_next_sn(last_sn, ):
    # print(f'last: {last_sn}')
    n = len(last_sn)
    predicted_4 = 2 * last_sn[n - 1] - last_sn[n - 2] + 29281
    return ctypes.c_uint32(predicted_4).value
    # return predicted_4

    # fig, ax = plt.subplots()
    # ax.plot(last_sn, label='last_sn')
    # ax.plot([len(last_sn)], [y_int], marker='o', markersize=3, color="red", label='predicted')
    # ax.plot([len(last_sn)], [expected], marker='o', markersize=3, color="green", label='actual')
    # ax.legend()


def plotly_double_plot(seq_numbers, x, y, z):
    # Create subplot figure
    fig = make_subplots(rows=2, cols=1, subplot_titles=['SN Plot', 'Vector Scatter Plot'])

    # Plot SN itself
    fig.add_trace(go.Scatter(x=list(range(len(seq_numbers))), y=seq_numbers, mode='lines', name='SN'), row=1, col=1)

    # Plot the vectors
    fig.add_trace(go.Scatter(x=list(range(len(x))), y=x, mode='markers', name='x[n] = s[n - 2] - s[n - 3]'), row=2,
                  col=1)
    fig.add_trace(go.Scatter(x=list(range(len(y))), y=y, mode='markers', name='y[n] = s[n - 1] - s[n - 2]'), row=2,
                  col=1)
    fig.add_trace(go.Scatter(x=list(range(len(z))), y=z, mode='markers', name='z[n] = s[n] - s[n - 1]'), row=2, col=1)

    # Update layout
    fig.update_layout(height=600, width=800, title_text="SN and Vector Scatter Plot", showlegend=True)

    # Update x-axis and y-axis labels
    fig.update_xaxes(title_text="Index (n)", row=1, col=1)
    fig.update_xaxes(title_text="Index (n)", row=2, col=1)

    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)

    # Show the figure
    fig.show()


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

    if PLOT_OR_PREDICT == 'plot':
        # Plot SN and vectors in subplots
        plt.figure(figsize=(15, 10))

        # Plot SN itself
        plt.subplot(2, 1, 1)
        plt.plot(seq_numbers, label='SN')
        plt.legend()
        plt.title('SN Plot')
        plt.xlabel('Index (n)')
        plt.ylabel('Value')

        # Plot the vectors
        plt.subplot(2, 1, 2)
        plt.scatter(range(len(x)), x, label='x[n] = s[n - 2] - s[n - 3]')
        plt.scatter(range(len(y)), y, label='y[n] = s[n - 1] - s[n - 2]')
        plt.scatter(range(len(z)), z, label='z[n] = s[n] - s[n - 1]')
        plt.legend()
        plt.title('Vector Scatter Plot')
        plt.xlabel('Index (n)')
        plt.ylabel('Value')

        plt.tight_layout()  # Adjust layout for better appearance
        # save to file 300 dpi
        plt.savefig('seq_numbers_and_x_y_z.png', dpi=300)

        plotly_double_plot(seq_numbers, x, y, z)

        # Plot the vectors in 3D space
        # fig = plt.figure(figsize=(10, 6))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x, y, z, label='Vectors', c='r', marker='o')
        #
        # # Set labels and title
        # ax.set_xlabel('x[n] = s[n - 2] - s[n - 3]')
        # ax.set_ylabel('y[n] = s[n - 1] - s[n - 2]')
        # ax.set_zlabel('z[n] = s[n] - s[n - 1]')
        # ax.set_title('3D Vector Plot')
        #
        # plt.legend()
        # plt.show()

        # Plot in each 2D combination
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # XY plane
        axes[0].scatter(x, y, c='r', marker='o')
        axes[0].set_xlabel('x[n] = s[n - 2] - s[n - 3]')
        axes[0].set_ylabel('y[n] = s[n - 1] - s[n - 2]')
        axes[0].set_title('XY Plane')

        # XZ plane
        axes[1].scatter(x, z, c='g', marker='o')
        axes[1].set_xlabel('x[n] = s[n - 2] - s[n - 3]')
        axes[1].set_ylabel('z[n] = s[n] - s[n - 1]')
        axes[1].set_title('XZ Plane')

        # YZ plane
        axes[2].scatter(y, z, c='b', marker='o')
        axes[2].set_xlabel('y[n] = s[n - 1] - s[n - 2]')
        axes[2].set_ylabel('z[n] = s[n] - s[n - 1]')
        axes[2].set_title('YZ Plane')

        plt.show()

        # Create a DataFrame for Plotly
        data = {'x': x, 'y': y, 'z': z}
        df = pd.DataFrame(data)
        # Create an interactive 3D scatter plot
        fig = px.scatter_3d(df, x='x', y='y', z='z', title='Interactive 3D Vector Plot',
                            labels={'x': 'X-axis', 'y': 'Y-axis', 'z': 'Z-axis'})
        fig.show()

    if PLOT_OR_PREDICT == 'predict':
        seq_numbers = seq_numbers[44900:]
        plot_list = []
        predicted_successfully = 0
        predictions_total = 0

        np_predicted = np.zeros(len(seq_numbers))
        # for i in range(2, len(seq_numbers) - 1):
        #     print("diff current and previous: ", seq_numbers[i] - seq_numbers[i - 1])
        for i in range(15, len(seq_numbers) - 1):
            LAST_N = 2
            print(f"Batch: [{i - LAST_N}:{i - 1}); Predicting {i}")

            last_n_sn = seq_numbers[i - LAST_N:i]
            last_n_x = x[i - LAST_N:i]
            last_n_y = y[i - LAST_N:i]
            last_n_z = z[i - LAST_N:i]
            predicted = predict_next_sn(
                last_sn=last_n_sn,
            )
            if predicted == seq_numbers[i]:
                predicted_successfully += 1
            print("Predicted: ", predicted, "Actual: ", seq_numbers[i], "Absolute Error: ",
                  predicted - seq_numbers[i])
            np_predicted[i] = predicted == seq_numbers[i]
            if not np_predicted[i]:
                print("\t Value in C ", ctypes.c_int32(predicted).value)
                print("\t Incorrect prediction (above)")

            predictions_total += 1
            plot_list.append([last_n_sn, seq_numbers[i], predicted])

        print(f'Predicted successfully: {predicted_successfully}')
        print(f'Predictions total: {predictions_total}')
        print(f'Accuracy: {(predicted_successfully / predictions_total) * 100}%')

        plot_sequence_numbers(seq_numbers, np_predicted)
        # Create a 3x3 subplot grid
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))

        # Iterate over the plot_list and populate subplots
        for i in range(3):
            for j in range(3):
                data = plot_list[i * 3 + j]

                last_n_sn = data[0]
                seq_number = data[1]
                predicted = data[2]

                # Plot last_n_sn
                axs[i, j].plot(last_n_sn, label='Last 3 SN')

                # Plot predicted and actual values
                axs[i, j].plot([len(last_n_sn)], [predicted], marker='o', markersize=10, color="red", label='Predicted',
                               alpha=0.5)
                axs[i, j].plot([len(last_n_sn)], [seq_number], marker='o', markersize=10, color="green", label='Actual',
                               alpha=0.5)

                axs[i, j].legend()
                axs[i, j].set_title(f'Plot {i * 3 + j + 1}. Correct Prediction: {predicted == seq_number}')

        # Adjust layout to prevent clipping
        plt.tight_layout()

        # save plot to file
        plt.savefig('predicted_values.png', dpi=300)
