from transformers import BertForSequenceClassification, BertTokenizer
import torch

model_path = "./best_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Move the model to the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set the model to evaluation mode
model.eval()

def predict_sentiment(aspect, sentence):
    # Combine aspect and sentence
    combined = aspect + " " + sentence
    
    # Tokenize the input
    inputs = tokenizer.encode_plus(
        combined,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move inputs to the same device as the model
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        
    # Get the predicted class
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    
    # Map the predicted class to the sentiment label
    sentiment_labels = ['conflict', 'negative', 'neutral', 'positive']
    predicted_sentiment = sentiment_labels[predicted_class]
    
    return predicted_sentiment

while True:
    sentence = input("Enter the sentence (or 'quit' to exit): ")
    if sentence.lower() == 'quit':
        break
    aspect = input("Enter the aspect: ")
    
    predicted_sentiment = predict_sentiment(aspect, sentence)
    print(f"Predicted sentiment: {predicted_sentiment}\n")