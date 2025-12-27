README
# Exam Score Prediction using Random Forest

A machine learning project that predicts student exam scores based on various factors including study habits, attendance, sleep patterns, and demographic information.

## 📋 Overview

This project uses a Random Forest Regressor to predict student exam scores. The model analyzes multiple features such as study hours, class attendance, sleep quality, and other factors to make accurate predictions.

## 🎯 Features

- **Data Preprocessing**: Handles missing values and encodes categorical variables
- **Feature Engineering**: Uses 11 different features for prediction
- **Machine Learning Model**: Random Forest Regressor with 200 estimators
- **Performance Metrics**: Evaluates model using R-squared score
- **Train-Test Split**: 90% training, 10% testing for robust evaluation

## 📊 Dataset

The dataset (`Exam_Score_Prediction.csv`) includes the following features:

- **age**: Student's age
- **gender**: Student's gender
- **course**: Course enrolled
- **study_hours**: Hours spent studying
- **class_attendance**: Attendance percentage
- **internet_access**: Internet availability
- **sleep_hours**: Hours of sleep
- **sleep_quality**: Quality of sleep
- **study_method**: Method of studying
- **facility_rating**: Rating of study facilities
- **exam_difficulty**: Difficulty level of exam
- **exam_score**: Target variable (what we're predicting)

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/exam-score-prediction.git
cd exam-score-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Place your `Exam_Score_Prediction.csv` file in the project directory

### Usage

Run the prediction script:
```bash
python predict_exam_scores.py
```

The script will:
1. Load and preprocess the data
2. Display the first few rows of the dataset
3. Check for missing values
4. Train the Random Forest model
5. Output the R-squared score

## 📈 Model Performance

The model uses:
- **Algorithm**: Random Forest Regressor
- **Number of Trees**: 200
- **Test Size**: 10% of data
- **Random State**: 42 (for reproducibility)

Expected R-squared score: *Run the model to see your results*

## 🛠️ Technologies Used

- **Python**: Programming language
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **Seaborn**: Data visualization
- **Matplotlib**: Plotting library

## 📁 Project Structure

```
exam-score-prediction/
│
├── predict_exam_scores.py    # Main prediction script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── Exam_Score_Prediction.csv  # Dataset (not included)
```

## 🔧 Customization

You can modify the model parameters in the code:

```python
# Change number of trees
model = RandomForestRegressor(n_estimators=300, random_state=42)

# Adjust train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


## 🙏 Acknowledgments

- Dataset source: On Kaggle By Kundan Sagar Bedmutha 
- Inspired by student performance analysis research
- Thanks to the scikit-learn community

---

⭐ If you found this project helpful, please give it a star!