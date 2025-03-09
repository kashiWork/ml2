from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

# Use the smaller T5 model for summarization.
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define a model for the incoming JSON payload.
class ReviewPayload(BaseModel):
    reviews: str

@app.post("/summarize")
def summarize_reviews(payload: ReviewPayload):
    reviews = payload.reviews
    if not reviews:
        raise HTTPException(status_code=400, detail="No reviews provided.")
    
    # T5 requires a prompt for summarization.
    input_text = "summarize: " + reviews
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate the summary.
    summary_ids = model.generate(input_ids, max_length=50, num_beams=2, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return {"summary": summary}
