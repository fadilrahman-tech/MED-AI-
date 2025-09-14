# MED-AI-
A healthcare assistant that predicts diabetes risk and provides a chatbot for medical FAQs, appointment booking, and insurance information.

# MedAI Chatbot Expert System

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
