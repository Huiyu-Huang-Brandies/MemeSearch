import torch
from PIL import Image
import timm
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class FeatureExtractor:
    def __init__(self, modelname):
        # Load the pre-trained model
        self.model = timm.create_model(
            modelname, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.model.eval()

        # Set up model input requirements
        config = resolve_data_config({}, model=self.model)
        self.preprocess = create_transform(**config)

    def extract_features(self, image_path):
        # Load and preprocess image
        input_image = Image.open(image_path)

        # Check for palette mode and convert if necessary
        if input_image.mode == "P":
            input_image = input_image.convert("RGBA")

        input_image = input_image.convert(
            "RGB"
        )  # Standardize to RGB for preprocessing
        input_image = self.preprocess(input_image)
        input_tensor = input_image.unsqueeze(0)

        # Perform inference and normalize features
        with torch.no_grad():
            feature_vector = self.model(input_tensor).squeeze().numpy()
        return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()
