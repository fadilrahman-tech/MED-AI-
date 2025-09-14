<img width="1365" height="601" alt="image" src="https://github.com/user-attachments/assets/a8b6b80a-d57e-4dec-b874-351ae7a425a8" />

# MedAI Chatbot Expert System <img width="265" height="213" alt="image" src="https://github.com/user-attachments/assets/e44ce041-11dd-4c6c-99b6-1d4f53c88e18" />


This project is a medical chatbot expert system designed to help users with preliminary health inquiries. It uses a predictive model to assess the likelihood of diabetes based on user-provided data.

## Features

- **Interactive Chatbot:** A user-friendly interface built with Streamlit for seamless interaction.
- **Diabetes Prediction:** A machine learning model (Random Forest Classifier) that analyzes health metrics to provide a prediction.
- **Knowledge Base:** The chatbot can answer common questions related to health and project functionality.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Joblib

## How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone [your-repository-url]
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd MedAI-Chatbot-Project
    ```
3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the Streamlit app:**
    ```bash
    streamlit run chat.py
    ```
