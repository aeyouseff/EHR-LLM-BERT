from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

data = pd.read_csv("data-ori.csv")
data = data.dropna()

def convert_to_text(row):
    return (
        f"The patient has HAEMATOCRIT={row['HAEMATOCRIT']}, "
        f"HAEMOGLOBINS={row['HAEMOGLOBINS']}, "
        f"ERYTHROCYTE={row['ERYTHROCYTE']}, "
        f"LEUCOCYTE={row['LEUCOCYTE']}, "
        f"THROMBOCYTE={row['THROMBOCYTE']}, "
        f"MCH={row['MCH']}, MCHC={row['MCHC']}, MCV={row['MCV']}, "
        f"AGE={row['AGE']}, SEX={row['SEX']}. "
        f"Is the patient in care ('in') or out care ('out')?"
    )

data['text'] = data.apply(convert_to_text, axis=1)
data['label'] = data['SOURCE'].apply(lambda x: 0 if x == "in" else 1)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42 )

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()

results = trainer.evaluate()
print(results)

trainer.save_model("EHR-BERT-MODEL")

test_texts = [
    "The patient has HAEMATOCRIT=38.2, HAEMOGLOBINS=12.7, ERYTHROCYTE=4.85, LEUCOCYTE=7.4, "
    "THROMBOCYTE=350, MCH=26.2, MCHC=34.5, MCV=80.2, AGE=30, SEX=M. "
    "Is the patient in care ('in') or out care ('out')?"
]

inputs = tokenizer(test_texts, truncation=True, padding=True, return_tensors="pt").to("cuda")

outputs = model(**inputs)
prediction = outputs.logits.argmax(dim=-1).item()

print("Prediction:", "in care" if prediction == 0 else "out care")
