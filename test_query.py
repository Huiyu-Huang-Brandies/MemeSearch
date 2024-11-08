from feature_extractor import FeatureExtractor
from milvus_manager import MilvusManager
from main_script import query_single_image

# Initialize feature extractor and Milvus manager
extractor = FeatureExtractor("efficientnet_b0")
milvus_manager = MilvusManager(
    uri="example.db", collection_name="meme_embeddings"
)

# Specify the path to your test image
test_image_path = "../tutorail/test/killer_whale/n02071294_20475.JPEG"

# Call the function to query this image
query_single_image(test_image_path)
