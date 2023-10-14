import torch
from model import HateModel
from data import DataModule


class HatePredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = HateModel.load_from_checkpoint(model_path,map_location="cpu")
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.lables = ["Normal Speech", "Toxic Speech"]

    def predict(self, text):
        inference_sample = {"comment_text": text}
        processed = self.processor.tokenize_data(inference_sample)
        logits = self.model(
            torch.tensor([processed["input_ids"]]).to(self.model.device),
            torch.tensor([processed["attention_mask"]]).to(self.model.device),
        )
        scores = self.softmax(logits[0]).tolist()
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions


if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = HatePredictor("./models/hate_model.ckpt")
    print(predictor.predict(sentence))

