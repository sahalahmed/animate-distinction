from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
from scipy.special import softmax

# dataset = load_dataset("yelp_review_full")
# print(dataset)

# Initialize an empty list to store data
my_list = []

# Open and read from 'human.txt' file, and append each line as a dictionary to 'my_list' with label 1
with open('human.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        my_list.append({"label": 1, "text": line})

# Open and read from 'concrete.txt' file, and append each line as a dictionary to 'my_list' with label 0
with open('concrete.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        my_list.append({"label": 0, "text": line})

# Create a Dataset object from the list
dataset = Dataset.from_list(my_list)

# Split the dataset into training and testing datasets with a test size of 0.3 (30%)
complete_ds = dataset.train_test_split(test_size=0.3, seed=20)

# Initialize a tokenizer with the BERT-base-cased model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Initialize a BERT-based model for sequence classification with 2 labels
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

# Define a function to tokenize the text data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize the datasets using the tokenizer
tokenized_datasets = complete_ds.map(tokenize_function, batched=True)

# Define training arguments including the output directory and evaluation strategy
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

# Load the accuracy metric
metric = evaluate.load("accuracy")

# Define a function to compute evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Initialize a Trainer object for training the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()




# print(trainer.predict(tokenized_datasets["test"]))

# Make predictions on the test dataset
predictions = trainer.predict(tokenized_datasets["test"])
logits = predictions.predictions

# Extract predicted labels
predicted_labels = np.argmax(predictions.predictions, axis=-1)

# Get the original labels from the test dataset
true_labels = tokenized_datasets["test"]["label"]

# Find indices where predictions are incorrect
correct_indices = np.where(predicted_labels == true_labels)[0]

# Print sentences that the model predicts wrong
result = []
print("Sorted Correct Predictions:")
for idx in correct_indices:
    probabilities = softmax(logits[idx])
    result.append({
        "true_label": true_labels[idx],
        "predicted_label": predicted_labels[idx],
        "probability": max(probabilities),
        "text": tokenized_datasets['test']['text'][idx]
    })
result_sorted = sorted(result, key=lambda x: x["probability"], reverse=True)

with open("sorted_human.txt", "w") as sorted_human:
    with open("sorted_concrete.txt", "w") as sorted_concrete:
        for entry in result_sorted:
            if entry['predicted_label'] == 1:
                sorted_human.write(f"True Label: {entry['true_label']}, Predicted Label: {entry['predicted_label']}, Probability: {entry['probability']}, Text: {entry['text']}\n")
            if entry['predicted_label'] == 0:
                sorted_concrete.write(f"True Label: {entry['true_label']}, Predicted Label: {entry['predicted_label']}, Probability: {entry['probability']}, Text: {entry['text']}\n")

# print(f"True Label: {entry['true_label']}, Predicted Label: {entry['predicted_label']}, Probability: {entry['probability']}, Text: {entry['text']}")