# prompt: create tkinter app (add number courses column)

import tkinter as tk
from tkinter import ttk
import joblib
import pandas as pd

def predict_marks():
        time_study = float(time_study_entry.get())
        num_courses = float(num_courses_entry.get())
        print(time_study)
        print(num_courses)

        # Load the saved model
        loaded_model = joblib.load('linear_regression_model.joblib')

        # Create a DataFrame with the input features
        input_data = pd.DataFrame({'number_courses': [num_courses],'time_study': [time_study], 'time_study_squared': [time_study**2]})

        # Make a prediction
        predicted_marks = loaded_model.predict(input_data)[0]

        result_label.config(text=f"Predicted Marks: {predicted_marks:.2f}")
    # except ValueError:
        # result_label.config(text="Invalid input. Please enter numbers.")
    # except FileNotFoundError:
        # result_label.config(text="Model file not found. Please train the model first.")


# Create main window
window = tk.Tk()
window.title("Student Marks Predictor")

# Time Study input
time_study_label = ttk.Label(window, text="Time Study (hours):")
time_study_label.grid(row=0, column=0, padx=5, pady=5)
time_study_entry = ttk.Entry(window)
time_study_entry.grid(row=0, column=1, padx=5, pady=5)

# Number of Courses input
num_courses_label = ttk.Label(window, text="Number of Courses:")
num_courses_label.grid(row=1, column=0, padx=5, pady=5)
num_courses_entry = ttk.Entry(window)
num_courses_entry.grid(row=1, column=1, padx=5, pady=5)


# Predict button
predict_button = ttk.Button(window, text="Predict", command=predict_marks)
predict_button.grid(row=2, column=0, columnspan=2, pady=10)

# Result label
result_label = ttk.Label(window, text="")
result_label.grid(row=3, column=0, columnspan=2)

window.mainloop()
