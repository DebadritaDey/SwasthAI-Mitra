import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np
import ast

# --- LOAD DATA AND MODELS ---
try:
    # Load the trained SVM model
    with open('svc.pkl', 'rb') as model_file:
        svc = pickle.load(model_file)

    # Load the datasets
    symtoms_df = pd.read_csv('symtoms_df.csv')
    precautions_df = pd.read_csv('precautions_df.csv')
    workout_df = pd.read_csv('workout_df.csv')
    description = pd.read_csv('description.csv')
    medications_df = pd.read_csv('medications.csv')
    diets_df = pd.read_csv('diets.csv')
    symptom_severity = pd.read_csv('Symptom-severity.csv')
    training_data = pd.read_csv('Training.csv')
    
    # Clean the column names to remove any extra whitespace
    training_data.columns = training_data.columns.str.strip()


except FileNotFoundError as e:
    st.error(f"Error loading files: {e}. Please make sure all the required CSV and PKL files are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()


# --- DATA PREPROCESSING & HELPER FUNCTIONS ---

# Get all symptoms directly from the trained model's features
all_symptoms = list(svc.feature_names_in_)

# Label encode the prognosis column from the training data
le = LabelEncoder()
le.fit(training_data['prognosis'])

def helper(disease):
    """
    Retrieves description, precautions, medications, diets, and workouts for a given disease.
    """
    try:
        desc = description[description['Disease'] == disease]['Description'].values[0]

        pre = precautions_df[precautions_df['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten().tolist()
        # Filter out nan values
        pre = [p for p in pre if pd.notna(p)]


        med = ast.literal_eval(medications_df[medications_df['Disease'] == disease]['Medication'].values[0])
        die = ast.literal_eval(diets_df[diets_df['Disease'] == disease]['Diet'].values[0])
        wrkout = workout_df[workout_df['disease'] == disease]['workout'].tolist()

        return desc, pre, med, die, wrkout
    except IndexError:
        return "Description not found.", [], [], [], []
    except Exception as e:
        st.error(f"An error occurred in the helper function: {e}")
        return "Error retrieving details.", [], [], [], []


def create_symptom_vector(symptoms, all_symptoms_list):
    """
    Creates a binary vector for the selected symptoms.
    """
    symptom_vector = np.zeros(len(all_symptoms_list))
    for symptom in symptoms:
        if symptom in all_symptoms_list:
            index = all_symptoms_list.index(symptom)
            symptom_vector[index] = 1
    return symptom_vector.reshape(1, -1)


# --- STREAMLIT UI ---

st.set_page_config(page_title="SwasthAIMitra", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        border: none;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stMultiSelect > div > div > div:nth-child(2) {
        background-color: white;
    }
    h1, h2, h3 {
        color: #1E3A8A; /* A deep blue color */
    }
    .recommendation-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .recommendation-card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .recommendation-title {
        font-size: 20px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 10px;
    }
    ul {
        list-style-type: '⚕️ ';
        padding-left: 20px;
    }
</style>
""", unsafe_allow_html=True)


st.title("👨‍⚕️ SwasthAIMitra")
st.markdown("This application helps you to get recommendations for a potential disease based on your symptoms. Please select your symptoms from the list below.")

# Symptom selection
selected_symptoms = st.multiselect(
    "Select your symptoms:",
    options=all_symptoms,
    help="You can select multiple symptoms."
)

if st.button("Get Recommendation"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        with st.spinner('Analyzing your symptoms...'):
            # Create input vector for the model
            symptom_vector = create_symptom_vector(selected_symptoms, all_symptoms)

            # Predict the disease
            prediction_index = svc.predict(symptom_vector)[0]
            predicted_disease = le.inverse_transform([prediction_index])[0]

            # Get recommendations
            description_text, precautions_list, medications_list, diet_plan_list, workout_plan_list = helper(predicted_disease)

            # Display the results
            st.success(f"**Predicted Disease: {predicted_disease}**")
            st.markdown("---")

            # Description
            st.header("Disease Description")
            st.write(description_text)

            col1, col2 = st.columns(2)

            with col1:
                with st.container():
                    st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                    st.markdown('<p class="recommendation-title">💊 Medications</p>', unsafe_allow_html=True)
                    if medications_list:
                        st.markdown("".join([f"<li>{med}</li>" for med in medications_list]), unsafe_allow_html=True)
                    else:
                        st.write("No specific medications found.")
                    st.markdown('</div>', unsafe_allow_html=True)

                with st.container():
                    st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                    st.markdown('<p class="recommendation-title">🥗 Diet Plan</p>', unsafe_allow_html=True)
                    if diet_plan_list:
                        st.markdown("".join([f"<li>{diet}</li>" for diet in diet_plan_list]), unsafe_allow_html=True)
                    else:
                        st.write("No specific diet plan found.")
                    st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                with st.container():
                    st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                    st.markdown('<p class="recommendation-title">⚠️ Precautions</p>', unsafe_allow_html=True)
                    if precautions_list:
                        st.markdown("".join([f"<li>{precaution}</li>" for precaution in precautions_list]), unsafe_allow_html=True)
                    else:
                        st.write("No specific precautions found.")
                    st.markdown('</div>', unsafe_allow_html=True)


                with st.container():
                    st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                    st.markdown('<p class="recommendation-title">💪 Workout Plan</p>', unsafe_allow_html=True)
                    if workout_plan_list:
                        st.markdown("".join([f"<li>{workout}</li>" for workout in workout_plan_list]), unsafe_allow_html=True)
                    else:
                        st.write("No specific workout plan found.")
                    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.info("**Disclaimer:** This is a recommendation system and not a substitute for professional medical advice. Please consult a doctor for an accurate diagnosis and treatment.")

