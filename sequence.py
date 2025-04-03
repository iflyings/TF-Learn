# echo $https_proxy
proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
}

from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, proxies=proxies)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, proxies=proxies)

sequence = "I've been waiting for a HuggingFace course my whole life."

input_ids = tokenizer(sequence, return_tensors="pt")
input_ids = input_ids
# tokens = tokenizer.tokenize(sequence)
# ids = tokenizer.convert_tokens_to_ids(tokens)
# input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)
