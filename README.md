# SOFE3370-Final-Project-Group-13
**Group 13**
+ Drew Glover - 100914229
+ Reid Hillis - 100915844
+ Jayden Mallari - 100927961
+ Aaraan Mahmood - 100872040
+ Jeffrey Atunure - 100880672

# 🔋 Battery Pack SOH Predictor with Gemini AI Chatbot

**SOFE 3370 Final Project - Group 13**  
*Intelligent Battery State of Health Prediction System*

---

## 📋 Project Overview

This project implements a machine learning-based Battery Pack State of Health (SOH) prediction system with an integrated AI chatbot. The system uses Linear Regression to predict battery health from voltage measurements (U1-U21) and provides an interactive Streamlit web interface for real-time predictions and AI-assisted analysis.

### Key Features
- **ML-Powered SOH Prediction**: Linear regression model trained on battery voltage data
- **Interactive Web Interface**: User-friendly Streamlit application
- **AI Chatbot Integration**: Google Gemini AI for intelligent battery health discussions
- **Real-Time Visualization**: Dynamic SOH gauge, performance metrics, and prediction plots
- **Batch Processing**: Evaluate entire datasets with comprehensive metrics (R², MSE, MAE)
- **Manual & File Input**: Support for CSV/XLSX uploads and manual data entry

---

## 🎯 What is State of Health (SOH)?

**State of Health (SOH)** is a critical metric that measures a battery's current capacity relative to its original rated capacity, expressed as a percentage or decimal (0.0 to 1.0). 

- **SOH = 1.0 (100%)**: Battery is in perfect condition
- **SOH < 0.8 (80%)**: Battery shows signs of degradation
- **SOH < 0.6 (60%)**: Battery requires attention or replacement

SOH prediction is essential for:
- Electric vehicle battery management
- Energy storage system monitoring
- Predictive maintenance scheduling
- Battery lifecycle optimization

---

## 🧠 Model Architecture & Performance

### Algorithm: Linear Regression
- **Input Features**: 21 voltage measurements (U1 through U21)
- **Target Variable**: Battery SOH
- **Training Split**: 80% training, 20% testing
- **Data Sorting**: Merge sort algorithm (O(n log n)) for SOH-based sorting

### Model Performance
Based on the PulseBat Dataset:
- **R² Score**: ~0.95+ (varies by run)
- **Mean Squared Error (MSE)**: < 0.01
- **Mean Absolute Error (MAE)**: < 0.05

*Note: Exact metrics depend on the random seed and test split.*

### Why Linear Regression?
- Fast training and prediction
- Interpretable coefficients for each voltage sensor
- Excellent performance on linearly correlated battery data
- Low computational overhead for deployment

---

## 📊 Dataset

**Source**: PulseBat Dataset  
**Format**: Excel (.xlsx) with 22 columns
- **Features**: U1, U2, U3, ..., U21 (individual cell voltages)
- **Target**: SOH (State of Health)

The dataset contains battery pack measurements across various charge/discharge cycles, enabling accurate SOH prediction based on voltage patterns.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd AlgorithmsFinalProject
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train the Model (Optional)
If you want to retrain the model:
```bash
python LinearRegression.py
```
This will:
- Load and sort the dataset using merge sort
- Train a linear regression model (80/20 split)
- Display performance metrics
- Save the trained model as `model.pkl`

### Step 4: Run the Application
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`

---

## 💻 Usage Guide

### Tab 1: SOH Prediction

#### Option A: Upload Dataset
1. Click "Browse files" and upload a CSV or XLSX file with U1-U21 columns
2. The app will automatically predict SOH for the first row
3. Check "Evaluate entire dataset" to run predictions on all rows
4. If an "Actual_SOH" column exists, performance metrics will be displayed

#### Option B: Manual Input
1. Check "Enter values manually"
2. Input voltage values (0.0 to 1.0) for U1 through U21
3. Adjust the "Healthy Threshold" slider (default: 0.60)
4. View predicted SOH and battery status

#### Visualization
- **SOH Gauge**: Horizontal bar showing predicted SOH percentage
- **Status Indicator**: Green (healthy) or Red (problem)
- **Scatter Plot**: Predicted vs Actual SOH (when actual values available)

### Tab 2: Gemini AI Chatbot

#### Setup
1. Enter your Google Gemini API key
2. Select a Gemini model from the dropdown (only appears after key entry)
3. Start chatting!

#### Chatbot Capabilities
- **Context-Aware**: Knows the latest SOH prediction and battery status
- **Dataset Analysis**: Can analyze uploaded dataset statistics
- **Battery Health Tips**: Provides maintenance and care recommendations
- **Conversational Memory**: Maintains chat history for context
- **Error Handling**: Falls back to static tips if API errors occur

#### Example Questions
- "What does this SOH value mean?"
- "How can I keep this battery healthy?"
- "What's the average voltage in my dataset?"
- "Should I replace this battery?"

---

## 📁 Project Structure

```
AlgorithmsFinalProject/
│
├── app.py                      # Main Streamlit application
├── LinearRegression.py         # Model training script
├── model.pkl                   # Trained Linear Regression model
├── PulseBat Dataset.xlsx       # Training dataset
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🔑 Getting a Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and paste it into the app
5. Ensure "Generative Language API" is enabled in your Google Cloud project

---

## 🛠️ Technologies Used

- **Python 3.x**: Core programming language
- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning library (Linear Regression)
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **google-generativeai**: Gemini AI integration
- **joblib**: Model serialization
- **openpyxl**: Excel file handling

---

## 🎓 Academic Context

**Course**: SOFE 3370 - Data Structures and Algorithms  
**Institution**: Ontario Tech University  
**Date**: Fall 2025  
**Group**: Group 13

### Project Highlights
- ✅ Implements merge sort algorithm for data preprocessing
- ✅ Demonstrates practical ML deployment
- ✅ Integrates cutting-edge generative AI
- ✅ Professional-grade error handling and UX
- ✅ Comprehensive documentation

---

## 🐛 Troubleshooting

### "Model not found" Error
- Ensure `model.pkl` exists in the project directory
- Run `LinearRegression.py` to generate the model

### Gemini API Errors
- Verify API key is valid and has quota remaining
- Check that "Generative Language API" is enabled
- Update library: `pip install --upgrade google-generativeai`
- Select a different model from the dropdown

### Dataset Upload Issues
- Ensure CSV/XLSX has columns named U1 through U21
- Check for missing values or non-numeric data
- Verify file encoding (UTF-8 recommended)

### Module Import Errors
```bash
pip install --upgrade -r requirements.txt
```

---

## 📸 Screenshots

### SOH Prediction Interface
![Add screenshot of Tab 1 showing gauge and prediction]

### Gemini Chatbot
![Add screenshot of Tab 2 showing chat interface]

### Performance Metrics
![Add screenshot showing R², MSE, MAE display]

---

## 🔮 Future Enhancements

- Support for other ML algorithms (Random Forest, Neural Networks)
- Real-time battery monitoring via IoT integration
- Historical SOH trend analysis and forecasting
- Multi-language support for international deployment
- Mobile-responsive design optimization
- Export prediction reports as PDF

---

## 📄 License

This project is developed for academic purposes as part of SOFE 3370 coursework.

---

## 🙏 Acknowledgments

- PulseBat Dataset providers
- Ontario Tech University Faculty
- Google Gemini AI team
- Streamlit community

---

**For questions or support, contact: [your-email@ontariotechu.net]**

*Last Updated: November 2025*

