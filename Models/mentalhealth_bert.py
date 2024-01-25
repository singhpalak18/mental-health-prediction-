pip install transformers 
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
from google.colab import files
upload= files.upload()
df=pd.read_excel('mental health dataset (1).xlsx')
df.head()
df.groupby(['Label']).size().plot.bar()
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
labels = {'anxiety':0,
          'depression':1,
          'frustration':2,
          'stress':3
          }

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [labels[label] for label in df['Label']]
        self.texts = [tokenizer(text,
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['Text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 4)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=10)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:

            model = model.to('cuda')
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()

                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()

            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()

                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc

            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | Train Accuracy: {total_acc_train / len(train_data): .3f} | Val Loss: {total_loss_val / len(val_data): .3f} | Val Accuracy: {total_acc_val / len(val_data): .3f}')

def evaluate(model, test_data):

    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=10)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)

              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
        return batch_texts, batch_y

np.random.seed(112)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                     [int(.8*len(df)), int(.9*len(df))])

print(len(df_train),len(df_val), len(df_test))

EPOCHS = 5
model = BertClassifier()
LR = 1e-5
train(model, df_train, df_val, LR, EPOCHS)

from google.colab import drive
drive.mount('/content/drive')
joblib.dump(tokenizer,'mentalhealth_tokenizer.joblib')
torch.save(model.state_dict(), 'mentalhealth_model.pth')
model=BertClassifier()
tokenizer = joblib.load('/content/drive/My Drive/mentalhealth_tokenizer.joblib')
model.load_state_dict(torch.load('/content/drive/My Drive/mentalhealth_model.pth',map_location='cpu'))

model.eval()
def evaluate_model(model, test_data):
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=10)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            preds = output.argmax(dim=1).cpu().numpy()  # Move predictions to CPU

            all_preds.extend(preds)
            all_labels.extend(test_label.cpu().numpy())  # Move labels to CPU

    # Calculate accuracy measures using all_preds and all_labels
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return accuracy, precision, recall, f1

# Call the function to calculate accuracy measures
accuracy, precision, recall, f1 = evaluate_model(model, df_test)

# Print the computed metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
def classify_response(question):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    encoded_question = tokenizer.encode_plus(
        question,
        max_length=256,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt'
    )
    input_ids = encoded_question['input_ids'].to('cuda')
    attn_mask = encoded_question['attention_mask'].to('cuda')

    model = BertClassifier()
    model = model.to('cuda')

    model.eval()
    with torch.no_grad():
        output = model(input_ids, attn_mask)

    probabilities = torch.softmax(output, dim=1)
    predicted_class_num = torch.argmax(probabilities, dim=1).item()

    class_names = ["Depression", "Anxiety", "Frustration", "Stress"]
    predicted_class_name = class_names[predicted_class_num]

    if question == "" or question.isspace():
        return "Please enter a valid answer!!"
    else:
        return predicted_class_name

questions = [
    "Do you ever worry about the security of your job?",
    "Do you feel confident at work?",
    "Have you started to avoid spending time with friends and loved ones due to extensive work?",
    "Do you feel like you have a good work-life balance here?",
    "Do you become irritable or annoyed more quickly than you have in the past?",
    "Do you often feel restless, on edge, or unable to relax?",
    "Do you feel comfortable talking about your mental health with others inside our organization?",
    "Is it difficult to fall asleep, get enough sleep, or wake up on time most days?",
    "Do you ever feel overworked or underworked here as an employee?",
    "Do you feel that your work is not recognized or underappreciated?"
]
user_responses = []
predicted_labels = []

for i, question in enumerate(questions):
    response = input(f"Q{i + 1}: {question} ")
    predicted_label = classify_response(response)  # Assuming this function is correctly defined
    predicted_labels.append(predicted_label)

print("Predicted Labels for Each Question:")
for i, label in enumerate(predicted_labels):
    print(f"Q{i + 1}: {label}")

label_counts = pd.Series(predicted_labels).value_counts()
labels = label_counts.index.tolist()
sizes = label_counts.values
colors=['purple', 'pink', 'cyan','red']

patches, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors, startangle=90, shadow=True, explode=(0.1,) * len(labels), autopct='%1.2f%%')
plt.axis('equal')
plt.title('Composition of Labels for All Questions')
plt.show()

autopct_values = [autotext.get_text().strip('%') for autotext in autotexts]
percentages = [float(value) for value in autopct_values]
depression_percentage = percentages[labels.index('Depression')]
anxiety_percentage = percentages[labels.index('Anxiety')]

pip install scikit-fuzzy 
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

depression_range = np.arange(0, 100, 1)
depression_var = ctrl.Antecedent(depression_range, 'Depression')
depression_var['low'] = fuzz.trimf(depression_range, [0, 0, 15])
depression_var['moderate'] = fuzz.trimf(depression_range, [15, 35, 40])
depression_var['high'] = fuzz.trimf(depression_range, [40, 100, 100])
extracted_depression_percentage = depression_percentage
low_membership = fuzz.interp_membership(depression_range, fuzz.trimf(depression_range, [0, 0, 15]), extracted_depression_percentage)
moderate_membership = fuzz.interp_membership(depression_range, fuzz.trimf(depression_range, [15, 35, 40]), extracted_depression_percentage)
high_membership = fuzz.interp_membership(depression_range, fuzz.trimf(depression_range, [40, 100, 100]), extracted_depression_percentage)
if moderate_membership > low_membership and moderate_membership > high_membership:
    severity_level = 'Moderate Depression'
elif high_membership > low_membership and high_membership > moderate_membership:
    severity_level = 'High Depression'
else:
    severity_level = 'Low Depression'
print("Severity Level of Depression:", severity_level)

anxiety_range = np.arange(0, 101, 1)
anxiety_var = ctrl.Antecedent(anxiety_range, 'Anxiety')
anxiety_var['low'] = fuzz.trimf(anxiety_range, [0, 0, 15])
anxiety_var['moderate'] = fuzz.trimf(anxiety_range, [15, 35, 40])
anxiety_var['high'] = fuzz.trimf(anxiety_range, [40, 100,100])
extracted_anxiety_percentage = anxiety_percentage
low_membership = fuzz.interp_membership(anxiety_range, fuzz.trimf(anxiety_range, [0, 0, 15]), extracted_anxiety_percentage)
moderate_membership = fuzz.interp_membership(anxiety_range, fuzz.trimf(anxiety_range, [15,35, 40]), extracted_anxiety_percentage)
high_membership = fuzz.interp_membership(anxiety_range, fuzz.trimf(anxiety_range, [40, 100, 100]), extracted_anxiety_percentage)
if moderate_membership > low_membership and moderate_membership > high_membership:
    severity_level = 'Moderate Anxiety'
elif high_membership > low_membership and high_membership > moderate_membership:
    severity_level = 'High Anxiety'
else:
    severity_level = 'Low Anxiety'
print("Severity Level of Anxiety:", severity_level)
