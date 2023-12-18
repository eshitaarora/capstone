---

# Air Quality Concentration Prediction using Ensemble Models

This repository contains a Streamlit application designed to predict air quality using ensemble machine learning models, including CNN, GRU, and LSTM. The application allows users to upload a dataset, select a target pollutant, and view the results of the prediction alongside actual values.

## Features

- Upload CSV dataset.
- Select target pollutant for prediction.
- Train ensemble models on the dataset.
- Visualize the comparison between actual and predicted values.
- (Optional) Predict future air quality for the next 30 days.

## Installation

To run this application, you need to install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Launch the application by running:

```bash
streamlit run air_quality.py
```

## Usage

1. Start the application.
2. Upload your CSV dataset using the file uploader.
3. Select the target pollutant from the dropdown.
4. Click on 'Train and Evaluate Models' to start the training process.
5. View the results and the graph comparing actual vs predicted values.
6. (Optional) Use the 'Predict Next 30 Days' feature to see future predictions.

## Requirements

This application requires the following modules:

- Streamlit
- Pandas
- NumPy
- PyTorch
- scikit-learn
- Matplotlib

See `requirements.txt` for specific version requirements.

## Data Format

The expected data format is a CSV file with columns representing different pollutants and environmental factors. Ensure the data is preprocessed appropriately before uploading.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](<link-to-your-issues-page>) for open issues or to create a new one.

## License

Distributed under the MIT License. See `LICENSE` for more information.

---
