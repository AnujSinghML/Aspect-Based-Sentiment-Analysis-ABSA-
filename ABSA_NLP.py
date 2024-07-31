#Training and Evaluating the Model

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import logging
import warnings
from transformers import logging as transformers_logging
import os

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.trainer")
# Set the checkpoint directory
checkpoint_dir = './results/checkpoint-624'

# Suppress specific warning
warnings.filterwarnings("ignore", message="A parameter name that contains `beta` will be renamed internally to `bias`.")

# Optionally, suppress all warnings from transformers
transformers_logging.set_verbosity_error()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
torch.manual_seed(42)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load the training and testing datasets
try:
    train_df = pd.read_csv('Restaurants_Train_v2.csv')
    test_df = pd.read_csv('restaurants-trial.csv')
except FileNotFoundError as e:
    logger.error(f"Error loading data: {e}")
    exit(1)

# Combine the aspect term with the sentence for feature extraction
train_df['Combined'] = train_df['Aspect Term'] + ' ' + train_df['Sentence']
test_df['Combined'] = test_df['Aspect Term'] + ' ' + test_df['Sentence']

# Get all unique classes from both train and test data
all_classes = sorted(set(train_df['polarity'].unique()) | set(test_df['polarity'].unique()))

# Encode the polarity labels
label_encoder = LabelEncoder()
label_encoder.fit(all_classes)
train_df['encoded_polarity'] = label_encoder.transform(train_df['polarity'])
test_df['encoded_polarity'] = label_encoder.transform(test_df['polarity'])

# Print out the classes
logger.info(f"Classes: {label_encoder.classes_}")
logger.info(f"Unique classes in training data: {train_df['polarity'].nunique()}")
logger.info(f"Unique classes in test data: {test_df['polarity'].nunique()}")

# Check class balance
logger.info(f"Class distribution in training set: {train_df['encoded_polarity'].value_counts(normalize=True)}")

# Split train data into train and validation
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['encoded_polarity'])

# Load the BERT tokenizer
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
except Exception as e:
    logger.error(f"Error loading tokenizer: {e}")
    exit(1)

class ABSADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __getitem__(self, index):
        sentence = self.data.Combined.iloc[index]
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        label = self.data.encoded_polarity.iloc[index]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def __len__(self):
        return self.len

# Prepare datasets
MAX_LEN = 128
train_dataset = ABSADataset(train_df, tokenizer, MAX_LEN)
val_dataset = ABSADataset(val_df, tokenizer, MAX_LEN)
test_dataset = ABSADataset(test_df, tokenizer, MAX_LEN)

# Load the BERT model for sequence classification
num_labels = len(label_encoder.classes_)
try:
    if os.path.exists(checkpoint_dir):
        logger.info(f"Loading model from checkpoint: {checkpoint_dir}")
        model = BertForSequenceClassification.from_pretrained(checkpoint_dir)
        if model.num_labels != num_labels:
            logger.warning(f"Number of labels in checkpoint ({model.num_labels}) doesn't match current data ({num_labels}). Adjusting model...")
            model.resize_token_embeddings(len(tokenizer))
            model.num_labels = num_labels
            model.classifier = torch.nn.Linear(model.config.hidden_size, num_labels)
    else:
        logger.info("Loading pre-trained BERT model")
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    model = model.to(device)
except Exception as e:
    logger.error(f"Error loading model: {e}")
    exit(1)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Define compute_metrics function for custom metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='macro')
    return {'macro_f1': f1}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
try:
    trainer.train(resume_from_checkpoint=checkpoint_dir if os.path.exists(checkpoint_dir) else None)
except Exception as e:
    logger.error(f"Error during training: {e}")
    exit(1)

# Save the best model
trainer.save_model("./best_model")

# Evaluate the model
try:
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    
    # Map predictions back to original labels
    pred_labels = label_encoder.inverse_transform(preds)
    true_labels = test_df['polarity']
    
    report = classification_report(true_labels, pred_labels)
    logger.info(f"Classification Report:\n{report}")
except Exception as e:
    logger.error(f"Error during evaluation: {e}")

logger.info("ABSA task completed successfully")

# Save the model and tokenizer
model.save_pretrained("./best_model")
tokenizer.save_pretrained("./best_model")