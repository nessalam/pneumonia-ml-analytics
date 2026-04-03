# Pneumonia Detection & Analytics Pipeline
Note: The notebook is pre-executed for easy viewing on GitHub.

## Overview
Developed a machine learning pipeline to classify chest X-ray images and built a SQL-based analytics system to monitor model performance.

## Results
- Achieved **~82% test accuracy**
- Trained on 2,400+ medical images
- Built SQL queries to analyze prediction accuracy and error patterns

## Tech Stack
- Python, TensorFlow, NumPy, Pandas
- SQL (SQLite)
- Data Visualization & Analytics

## Key Features
- Image classification model for pneumonia detection
- SQL database storing prediction results
- Queries to track:
  - Accuracy
  - Error rates
  - Confidence levels
- Structured pipeline from data → model → analytics

## Setup
pip install -r requirements.txt

## Example SQL Query
```sql
SELECT 
  AVG(CASE WHEN predicted_label = true_label THEN 1 ELSE 0 END) AS accuracy
FROM prediction_logs;
