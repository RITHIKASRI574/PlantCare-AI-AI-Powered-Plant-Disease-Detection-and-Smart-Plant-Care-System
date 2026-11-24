# 🌱 PlantCare AI – AI-Powered Plant Disease Detection and Smart Plant Care System

## About the Project

PlantCare AI is a project I developed to help farmers and plant lovers identify plant diseases more easily.

Plant diseases can affect crop growth and reduce agricultural productivity. Sometimes, farmers may not be able to identify a disease at an early stage or get immediate expert advice.

To address this problem, I built PlantCare AI using Python, Streamlit and a CNN-based deep learning model. Users can upload a picture of a plant leaf, and the application analyzes it to predict the possible disease. It also provides useful information about the disease, along with remedies and preventive measures.

My goal is to make basic plant disease information and plant care guidance easier to access through a simple web application.

## What Can PlantCare AI Do?

Here are the main features I have included in this project:

- **Plant Disease Detection:** Users can upload a plant leaf image and get an AI-based disease prediction.
- **Confidence Score:** The application displays the model's confidence score for its prediction.
- **Disease Information:** Provides details about the detected disease.
- **Remedies and Prevention:** Shows suggested ways to manage the disease and prevent further spread.
- **English and Tamil Chatbot:** Users can ask plant-related questions in either English or Tamil.
- **Voice Assistant:** Helps users access plant care information through text-to-speech.
- **Prediction History:** Saves previous prediction records so users can refer to them later.
- **Follow-up Reminders:** Reminds users to inspect their plants and monitor their recovery.

## Technologies I Used

| Technology | How I Used It |
|---|---|
| Python | Main programming language |
| Streamlit | Building the web application |
| TensorFlow | Working with the deep learning model |
| Keras | Building and training the CNN model |
| NumPy | Numerical operations |
| Pandas | Handling data |
| Pillow | Processing plant images |
| SQLite | Storing prediction history and application data |
| gTTS | Providing voice output |
| Git and GitHub | Managing and hosting the project |

## How the Application Works

The application follows a simple process:

1. The user uploads an image of a plant leaf.
2. The image is resized and preprocessed.
3. The trained CNN model analyzes the image.
4. The model predicts the possible plant disease.
5. The application displays the disease name and confidence score.
6. The user can view disease information, remedies and preventive suggestions.
7. The user can also use the chatbot, voice assistant and follow-up reminders.

**Basic workflow:**

Plant Leaf Image → Image Preprocessing → CNN Model → Disease Prediction → Remedies and Plant Care Suggestions

## Project Structure

```text
Plant-Disease-Detection/
│
├── app.py
├── train_model.py
├── preprocess.py
├── split_dataset.py
├── evaluate_model.py
├── verify_split.py
│
├── requirements.txt
├── .gitignore
│
├── model/
│   ├── plant_disease_model.keras
│   └── class_names.json
│
├── dataset/
├── dataset_split/
│
└── README.md
```

*Note: The dataset folders may not be included in the GitHub repository because of their large size.*

## How to Run the Project on Your Computer

If you want to try this project on your own computer, follow these steps.

### 1. Clone the Repository

```bash
git clone https://github.com/RITHIKASRI574/PlantCare-AI-AI-Powered-Plant-Disease-Detection-and-Smart-Plant-Care-System.git
```

### 2. Open the Project Folder

```bash
cd PlantCare-AI-AI-Powered-Plant-Disease-Detection-and-Smart-Plant-Care-System
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 4. Install the Required Packages

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

*Note: The trained model files must also be available in the `model` folder. The dataset is needed if you want to retrain the model.*

## Live Demo

I have also deployed PlantCare AI using Streamlit Community Cloud.

**Try the application here:**

[Add your Streamlit application link here]

## Screenshots

Here are some screenshots of my application.

- Home page
- Plant disease detection page
- Prediction results
- Disease information and remedies
- English and Tamil chatbot
- Plant history
- Follow-up reminders

*(I will add the application screenshots here.)*

## What I Would Like to Improve in the Future

There are still many things I would like to improve in this project.

Some of my future plans include:

- Adding support for more plant varieties and diseases.
- Improving the prediction performance of the CNN model.
- Supporting more regional languages.
- Making the application more useful for farmers with limited internet access.
- Improving plant recovery tracking.
- Adding more plant care information and features.

## A Small Note

This project is intended to provide preliminary plant disease information and general care suggestions. AI predictions may not always be correct, so users should consult agricultural experts before making important treatment decisions.

## About Me

I developed this project as part of my learning journey in Artificial Intelligence and Data Science.

While working on PlantCare AI, I explored Python, deep learning, CNN models, image processing, Streamlit and database integration. This project helped me understand how AI concepts can be applied to solve practical problems in agriculture.

**GitHub:** [RITHIKASRI574](https://github.com/RITHIKASRI574)

---

If you find this project useful, feel free to explore the repository and give it a ⭐.

Thank you for visiting PlantCare AI! 🌱
