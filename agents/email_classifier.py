import joblib
import numpy as np
from crewai import Agent

class EmailClassifierAgent:
    def __init__(self, model_path="models/model.pkl", vectorizer_path="models/vectorizer.pkl"):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.agent = Agent(
            name="EmailClassifier",
            role="Classifies internal emails",
            goal="Classify email into HR, IT, or Other",
            backstory="You are an expert in categorizing internal company emails into HR, IT, or Other for efficient workflow."
        )

    def run(self, input_text):
        X = self.vectorizer.transform([input_text])
        proba = self.model.predict_proba(X)[0]
        pred_index = np.argmax(proba)
        confidence = float(proba[pred_index])
        category = self.model.classes_[pred_index]
        return {
            "email_text": input_text,
            "predicted_category": category,
            "confidence": confidence
        }
