from src.models import *
import pickle
import torch


class_model_path = 'Classification/Galacticum/galacticum_cnn_milk_v1.pth'
reg_model_path = 'Regression/Galacticum/lama_galacticum_regression_v1.sav'
label_encoder_path = 'src/le.sav'



class Pipeline():
    def __init__(self, data):
        self.class_model = ClfModelTabCNN(
            input_dim=15600,
            output_dim=3, 
            ).load_state_dict(torch.load(class_model_path), map_location='cpu')

        self.reg_model = pickle.load(reg_model_path)
        self.label_encoder = pickle.load(label_encoder_path)

        self.X = data

    def get_classification(self, ):
        X = torch.tensor(self.X.values, dtype=torch.float)

        logits = self.class_model(X)
        prediction = logits.argmax(1).numpy()

        decoded_prediction = self.label_encoder.inverse_transform(prediction)[0]

        return decoded_prediction

    def get_regression(self, ):
        prediction = self.reg_model.predict(X)

        return prediction

    def get_predicted_data(self, ):

        tabular_pred = self.X
        tabular_pred['substance'] = self.get_classification()
        tabular_pred['target'] = self.get_regression()

        return tabular_pred
