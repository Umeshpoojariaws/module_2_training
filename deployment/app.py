import pickle
import pandas as pd
from fastapi import FastAPI

# Load the model artifact
MODEL_PATH = "model/ml_service.pkl"
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(passenger_count: int, trip_distance: float):
    # Prepare the input data structure matching training features
    data = pd.DataFrame([{'passenger_count': passenger_count, 'trip_distance': trip_distance}])

    # Make prediction
    prediction = model.predict(data)[0]

    return {"prediction": int(prediction)}

if __name__ == "__main__":
    # Simple run command for testing locally (not used in Docker typically)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9699)