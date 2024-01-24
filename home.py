import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from torch import nn
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
labels = {'anxiety': 0, 'depression': 1, 'frustration': 2, 'stress': 3}

class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 4)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

# Load PyTorch BERT model and tokenizer
def load_model():
    model = BertClassifier()
    model.load_state_dict(torch.load('Models/mentalhealth_model.pth', map_location=torch.device('cpu')))
    return model

model = load_model()
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Function to classify user response
def classify_response(question, model, tokenizer):
    encoded_question = tokenizer.encode_plus(
        question,
        max_length=256,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt'
    )
    input_ids = encoded_question['input_ids']
    attn_mask = encoded_question['attention_mask']

    model.eval()
    with torch.no_grad():
        output = model(input_ids, attn_mask)

    probabilities = torch.softmax(output, dim=1)
    predicted_class_num = torch.argmax(probabilities, dim=1).item()

    class_names = ["Depression", "Anxiety", "Frustration", "Stress"]
    return class_names[predicted_class_num]

# Streamlit app
st.title("Mental Health Prediction")

# List of questions
questions = [
    "How often have you felt that things were going your way?",
    "How often have you felt that you were on top of things?",
    "How often have you felt difficulties were piling up so high that you could not overcome them?",
    "How often do you want to spend time with your friends and family ?",
    "Do you find it hard to make decisions than before ?",
    "Do you find your work interesting or does it feel like you are being pushed to work ?",
    "Do you frequently experience a sense of restlessness or being on edge?",
    "Have you noticed an increase in irritability or a shorter fuse in your reactions?",
    "Do you notice a persistent feeling of tightness or tension in your muscles?",
    "Are you quick to criticize others?",
    "Do you get annoyed with yourself?"
]

# Display questions and collect responses
user_responses = []
predicted_labels = []
for i, question in enumerate(questions):
    st.markdown(f"**Q{i + 1}: {question}**")
    response = st.text_area(f"Your response to Q{i + 1}")
    user_responses.append(response)

    # Classify individual answers
    
    with st.spinner("Classifying..."):
        predicted_label = classify_response(response, model, tokenizer)
        predicted_labels.append(predicted_label)
        # st.write(predicted_label)
    st.markdown("---")

# Display composition graph at the end
st.title('Composition of Labels for All Questions')
st.markdown("---")

# Plot the pie chart using Matplotlib
label_counts = pd.Series(predicted_labels).value_counts()
labels = label_counts.index.tolist()
sizes = label_counts.values
colors = ['purple', 'pink', 'cyan', 'red']

fig, ax = plt.subplots()
patches, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, startangle=90, shadow=True,
                                   explode=(0.1,) * len(labels), autopct='%1.2f%%')

# Save the figure and display it in Streamlit
st.pyplot(fig)

# Extract the percentages
autopct_values = [text.get_text().strip('%') for text in autotexts]

# Converting extracted percentages to float
percentages = [float(value) for value in autopct_values]

# Storing percentages of Depression and Anxiety in variables
depression_percentage = percentages[labels.index('Depression')] if 'Depression' in labels else 0.0
anxiety_percentage = percentages[labels.index('Anxiety')] if 'Anxiety' in labels else 0.0

# Fuzzy Logic for Depression
depression_range = np.arange(0, 100, 1)
depression_var = ctrl.Antecedent(depression_range, 'Depression')
depression_var['low'] = fuzz.trimf(depression_range, [0, 0, 15])
depression_var['moderate'] = fuzz.trimf(depression_range, [15, 35, 40])
depression_var['high'] = fuzz.trimf(depression_range, [40, 100, 100])

# Calculate membership for extracted depression percentage
low_membership = fuzz.interp_membership(depression_range, fuzz.trimf(depression_range, [0, 0, 15]), depression_percentage)
moderate_membership = fuzz.interp_membership(depression_range, fuzz.trimf(depression_range, [15, 35, 40]), depression_percentage)
high_membership = fuzz.interp_membership(depression_range, fuzz.trimf(depression_range, [40, 100, 100]), depression_percentage)

# Determine severity level based on membership
if moderate_membership > low_membership and moderate_membership > high_membership:
    depression_severity = 'Moderate Depression'
elif high_membership > low_membership and high_membership > moderate_membership:
    depression_severity = 'High Depression'
else:
    depression_severity = 'Low Depression'

st.write("Severity Level of Depression:", depression_severity)

# Fuzzy Logic for Anxiety
anxiety_range = np.arange(0, 101, 1)
anxiety_var = ctrl.Antecedent(anxiety_range, 'Anxiety')
anxiety_var['low'] = fuzz.trimf(anxiety_range, [0, 0, 15])
anxiety_var['moderate'] = fuzz.trimf(anxiety_range, [15, 35, 40])
anxiety_var['high'] = fuzz.trimf(anxiety_range, [40, 100, 100])

# Calculate membership for extracted anxiety percentage
low_membership = fuzz.interp_membership(anxiety_range, fuzz.trimf(anxiety_range, [0, 0, 15]), anxiety_percentage)
moderate_membership = fuzz.interp_membership(anxiety_range, fuzz.trimf(anxiety_range, [15, 35, 40]), anxiety_percentage)
high_membership = fuzz.interp_membership(anxiety_range, fuzz.trimf(anxiety_range, [40, 100, 100]), anxiety_percentage)

# Determine severity level based on membership
if moderate_membership > low_membership and moderate_membership > high_membership:
    anxiety_severity = 'Moderate Anxiety'
elif high_membership > low_membership and high_membership > moderate_membership:
    anxiety_severity = 'High Anxiety'
else:
    anxiety_severity = 'Low Anxiety'

st.write("Severity Level of Anxiety:", anxiety_severity)

