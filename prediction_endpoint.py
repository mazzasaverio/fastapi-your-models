import contextlib
import os
import joblib
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from loguru import logger

# Basic logger configuration
logger.add("file_{time}.log", rotation="1 week")

class PredictionInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    category: str

class NewsgroupsModel:
    model: Pipeline | None = None
    targets: list[str] | None = None

    def load_model(self) -> None:
        """Loads the model"""
        logger.info("Loading the model")
        model_file = os.path.join(os.path.dirname(__file__), "newsgroups_model.joblib")
        try:
            loaded_model: tuple[Pipeline, list[str]] = joblib.load(model_file)
            self.model, self.targets = loaded_model
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

    async def predict(self, input: PredictionInput) -> PredictionOutput:
        """Runs a prediction"""
        logger.info(f"Running prediction for input: {input.text}")
        if not self.model or not self.targets:
            logger.warning("Model is not loaded")
            raise RuntimeError("Model is not loaded")
        try:
            prediction = self.model.predict([input.text])
            category = self.targets[prediction[0]]
            logger.info(f"Prediction successful: {category}")
            return PredictionOutput(category=category)
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

newgroups_model = NewsgroupsModel()

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application lifespan")
    newgroups_model.load_model()
    yield
    logger.info("Ending application lifespan")

app = FastAPI(lifespan=lifespan)

@app.post("/prediction")
async def prediction(
    output: PredictionOutput = Depends(newgroups_model.predict),
) -> PredictionOutput:
    return output
