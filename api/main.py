from fastapi import FastAPI
from model import get_prediction
from pydantic import BaseModel

class Review(BaseModel):
    """Define data model."""
    review_full: str

    class Config:
        """Define example schema."""
        schema_extra = {
            "example": {
                "review_full": "This restaurant is terrible. Don't go here.",
            }
        }

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "This is the CWT example task."}

@app.post("/predict/")
async def predict(review: Review):
    """Get predictions using request body."""
    review_dict = review.dict()
    return get_prediction([review_dict['review_full']])