from fastapi import FastAPI, HTTPException
import requests
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

app = FastAPI()

# Load the Pegasus model and tokenizer.
model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

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
    
    # Tokenize the input text (reviews), limiting to 512 tokens.
    input_ids = tokenizer.encode(reviews, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate the summary.
    summary_ids = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return {"summary": summary}
