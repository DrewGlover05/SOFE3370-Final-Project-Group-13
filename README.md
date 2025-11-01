# SOFE3370-Final-Project-Group-13
**Group 13**
+ Drew Glover - 100914229
+ Reid Hillis - 100915844
+ Jayden Mallari - 100927961
+ Aaraan Mahmood - 100872040
+ Jeffrey Atunure - 100880672

<br>**Set up instructions:**

1. Open the file in a Python IDE
2. Enter these 2 lines separately in the terminal to install the required libraries:

&emsp;&emsp;pip install numpy pandas scikit-learn matplotlib

&emsp;&emsp;pip install openpyxl


3. Before running the code, ensure that the whole file path referenced. The excel file must be in the same folder as the Linear Regression Model.

&emsp;&emsp;For Example:

&emsp;&emsp;df = pd.read_excel("C:\\FinalProjAlg\\PulseBat Dataset.xlsx")

4. Run the code, and the user is prompted with 'Enter SOH threshold value:', enter a threshold between 0-1.

&emsp;&emsp;For example:

&emsp;&emsp;Enter SOH threshold value: 0.7


Running The Chatbot:

5. In the terminal, type:

```
pip install io os google.generativeai sklearn.metrics streamlit
```

```
streamlit run app.py
```





Some caveats to worry about:

Battery Prediction Tab upload - Only accepts .xlsx file format

Gemini API Chatbot - 

Manually entering values in the battery prediction tab - Only between 0.0 - 4.1 (inclusive)

Developer's note *Important*: If you make any changes, you need to save and rerun the program from the terminal. Don't press the rerun button on the chatbot, the changes won't be applied.

