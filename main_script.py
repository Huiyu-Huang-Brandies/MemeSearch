import os
import random
from PIL import Image
from feature_extractor import FeatureExtractor
from milvus_manager import MilvusManager

# Initialize components
extractor = FeatureExtractor("efficientnet_b0")
milvus_manager = MilvusManager(
    uri="example.db", collection_name="meme_embeddings"
)

# Define folders
train_folder = "./train"
test_folder = "./test"
result_image_folder = "./results"


# Query test images and save results
def query_and_save_results(query_image_path, idx):
    query_image_resized_path = f"query_image_{idx}_resized.jpg"
    Image.open(query_image_path).resize((150, 150)).save(
        query_image_resized_path
    )

    query_vector = extractor.extract_features(query_image_path)
    results = milvus_manager.search_similar(query_vector)

    images = []
    for result in results:
        for hit in result:
            filename = hit["entity"]["filename"]
            img = Image.open(filename).resize((150, 150)).convert("RGB")
            images.append(img)

    # Create concatenated image for top 10 results
    width, height = 150 * 5, 150 * 2
    concatenated_image = Image.new("RGB", (width, height))
    for i, img in enumerate(images[:10]):
        x, y = i % 5, i // 5
        concatenated_image.paste(img, (x * 150, y * 150))

    result_image_path = os.path.join(
        result_image_folder, f"results_meme_image_{idx}.jpg"
    )
    concatenated_image.save(result_image_path)
    return query_image_resized_path, result_image_path


if __name__ == "__main__":
    # Insert training images into Milvus
    for filename in os.listdir(train_folder):
        if filename.endswith(".png"):
            filepath = os.path.join(train_folder, filename)
            image_embedding = extractor.extract_features(filepath)
            milvus_manager.insert_image(
                image_embedding, {"filename": filepath}
            )

    # Run search for multiple test images
    os.makedirs(result_image_folder, exist_ok=True)
    num_tests = 5
    test_images = random.sample(os.listdir(test_folder), num_tests)

    for idx, test_image in enumerate(test_images, start=1):
        query_image_path = os.path.join(test_folder, test_image)
        query_resized, result_path = query_and_save_results(
            query_image_path, idx
        )

        print(f"Query {idx}: {test_image}")
        print(f"Saved query image as {query_resized}")
        print(f"Top 1 similar images saved as {result_path}\n")


def query_single_image(image_path):
    # Replace slashes with underscores for the output filename
    normalized_name = image_path.replace("/", "_").replace("\\", "_")
    name, _ = os.path.splitext(normalized_name)  # Remove the file extension
    query_image_resized_path = f"./results/{name}_query_resized.jpg"
    top_image_path = f"./results/{name}_top_result.jpg"

    # Convert query image to correct size and ensure it's in RGB mode
    query_image = Image.open(image_path).resize((150, 150))

    # Convert to RGB if it's not already in RGB mode (JPEG is typically RGB)
    if query_image.mode not in ["RGB"]:
        query_image = query_image.convert("RGB")

    query_image.save(query_image_resized_path)

    # Extract features and perform search for top 1 result
    query_vector = extractor.extract_features(image_path)
    results = milvus_manager.search_similar(query_vector, top_k=1)

    # Check if results are valid
    if not results or not results[0]:
        print("No results found for the query image.")
        return

    # Retrieve and display the top result
    top_result = results[0][0]
    filename = top_result["entity"]["filename"]

    # Open and resize the top result image
    top_image = Image.open(filename).resize((150, 150))
    if top_image.mode not in ["RGB"]:
        top_image = top_image.convert("RGB")

    top_image.save(top_image_path)

    print(f"Query image saved as {query_image_resized_path}")
    print(f"Top result image saved as {top_image_path}\n")
