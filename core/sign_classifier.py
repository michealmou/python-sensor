import pickle
import os
from config import BASE_DIR

class SignClassifier:
    def __init__(self, model_path=None, labels_path=None):
        model_dir = os.path.join(BASE_DIR, "models", "saved")

        if model_path is None:
            model_path = os.path.join(model_dir, "model_rf.p")
        if labels_path is None:
            labels_path = os.path.join(model_dir, "labels.pickle")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(labels_path, "rb") as f:
            self.label_map = pickle.load(f)

    def predict(self, landmarks):
        """Predict sign from landmark features. Returns human-readable sign name."""
        prediction = self.model.predict([landmarks])[0]
        return self.label_map.get(prediction, str(prediction))
