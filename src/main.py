from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

MODEL_PATH = "src/model/wine_classification.pkl"  
SCALER_PATH = "src/model/scaler.joblib"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model or scaler: {e}")

class WineInput(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

app = FastAPI(title="Wine Classifier API", version="1.0")

@app.post("/predict")
def predict_wine(input: WineInput):
    try:
        input_data = np.array([[value for value in input.dict().values()]])
        input_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled).tolist()[0]
        
        return {
            "predicted_class": int(prediction),
            "probability": probability
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
