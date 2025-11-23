# advanced-time-series-forecasting-with-LSTMs
This repository  implementation of a sophisticated time series forecasting solution using an Encoder-Decoder Long Short-Term Memory (LSTM) network augmented with a custom Attention Mechanism. This project moves beyond traditional methods like ARIMA and simple RNNs to tackle complex, synthetic, multi-seasonal, and heteroscedastic time series data.
This repository contains a Jupyter Notebook implementing an **LSTM (Long Short-Term Memory) neural network** for time-series forecasting using **TensorFlow**. The project demonstrates data preprocessing, model training, evaluation, and visualization of predictions.

---
Attention-Based LSTM for Time Series Prediction
Overview
This Jupyter Notebook implements and trains a deep learning model for time series forecasting. The model utilizes an LSTM layer to capture temporal dependencies, coupled with a custom Attention Layer to selectively focus on the most relevant parts of the input sequence for making the final prediction.

Dependencies and Setup
The following Python libraries are required (installed within the notebook):

tensorflow (for the deep learning model)

numpy (for data manipulation)

pandas (for data frame operations)

matplotlib (for plotting and visualization)

scikit-learn (for data splitting and evaluation metrics)

The notebook automatically creates the following output directory structure:

outputs/logs

outputs/models

outputs/predictions

outputs/plots

Data
The dataset is synthetically generated using a sinusoidal function with added Gaussian noise.

Task: Predict the next value in the sequence based on the preceding 50 time steps.

Total Samples: 5,000

Sequence Length: 50 time steps (seq_len=50)

Splitting: The data is split into an 80% training set (4,000 samples) and a 20% validation set (1,000 samples).

Saved Files:

outputs/X_data.npy (Input features)

outputs/y_data.npy (Target values)

outputs/predictions/dataset_preview.csv (A small preview of the generated targets)

Model Architecture
The model is a sequential recurrent neural network with a custom attention mechanism:

Input Layer: Takes sequences of shape (50,).

ExpandDimsLayer: A custom layer to reshape the input to (50, 1) for the LSTM.

LSTM Layer: A recurrent layer with 64 units, returning sequences.

AttentionLayer: A custom layer that calculates a context vector by applying softmax-based attention weights to the LSTM output sequence. It outputs both the context vector and the attn_weights.

Output Layer: A final Dense(1) layer on the context vector for the prediction.

The model is compiled using the Adam optimizer and Mean Squared Error (MSE) loss.

Training and Results
The model is trained for 50 epochs using a batch size of 32.

Callbacks: Training uses ModelCheckpoint to save the best model based on validation loss, and EarlyStopping with a patience of 5 epochs.

Best Model: The best model is saved to outputs/models/best_model.h5.

Evaluation Metrics (on Validation Data)
After training, the model achieves the following performance metrics on the validation set:

Mean Squared Error (MSE): 0.0029

Mean Absolute Error (MAE): 0.0432

R-squared (R2 Score): 0.7610

Correlation (Actual vs. Predicted): 0.8829

Visualization Outputs
The notebook saves several plots to the outputs/plots/ directory, including:

Training loss curve.

Scatter plot of actual vs. predicted values.

A visualization of actual vs. predicted values over the validation set (val_predictions_visualization.png).

A stem plot showing the attention weights for a specific sample (attention_sample0.png).

