from fastapi import FastAPI, HTTPException
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

# Use the smaller T5 model for summarization.
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Function to fetch reviews from your Beeceptor mock server.
def get_mock_reviews():
    url = "https://dandelion1.free.beeceptor.com"  # Your mock API URL
    response = requests.get(url, headers={"some-header": "some-value"})
    if response.status_code == 200:
        data = response.json()  # Assuming the response is JSON
        reviews = " ".join([review["text"] for review in data["reviews"]])
        return reviews
    else:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch reviews.")

@app.get("/summarize")
def summarize_reviews():
    reviews = get_mock_reviews()
    if not reviews:
        raise HTTPException(status_code=400, detail="No reviews found.")
    
    # T5 requires a prompt for summarization.
    input_text = "summarize: " + reviews
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate the summary (using fewer beams can reduce memory usage).
    summary_ids = model.generate(input_ids, max_length=50, num_beams=2, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return {"summary": summary}
