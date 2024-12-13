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
test_folder = "./test"
result_image_folder = "./results"


# Query test images and save results
def query_and_save_results(query_image_path, idx, label_filter=None, top_k=4):
    # Save the original query image
    query_image_resized_path = f"query_image_{idx}.jpg"
    Image.open(query_image_path).convert("RGB").save(query_image_resized_path)

    query_vector = extractor.extract_features(query_image_path)
    results = milvus_manager.search_similar(query_vector, top_k=top_k)

    filtered_results = []
    for result in results:
        for hit in result:
            # Check if "label" exists before filtering
            if (
                label_filter is None
                or hit["entity"].get("label") == label_filter
            ):
                filename = hit["entity"]["filename"]
                img = Image.open(filename).convert("RGB")  # Keep original size
                filtered_results.append(img)

    if not filtered_results:
        print("No matching images found for the specified label filter.")
        return query_image_resized_path, None

    # Dynamically determine canvas size based on image dimensions
    max_width = max(img.width for img in filtered_results[:top_k])
    max_height = max(img.height for img in filtered_results[:top_k])

    # Create a canvas to display results side by side
    total_width = max_width * len(filtered_results[:top_k])
    canvas_height = max_height
    concatenated_image = Image.new("RGB", (total_width, canvas_height))

    # Paste images into the canvas
    for i, img in enumerate(filtered_results[:top_k]):
        concatenated_image.paste(img, (i * max_width, 0))

    result_image_path = os.path.join(
        result_image_folder, f"results_meme_image_{idx}.jpg"
    )
    concatenated_image.save(result_image_path)
    return query_image_resized_path, result_image_path


if __name__ == "__main__":
    os.makedirs(result_image_folder, exist_ok=True)
    num_tests = 5
    test_images = random.sample(os.listdir(test_folder), num_tests)

    for idx, test_image in enumerate(test_images, start=1):
        query_image_path = os.path.join(test_folder, test_image)
        query_resized, result_path = query_and_save_results(
            query_image_path, idx, label_filter=1  # Only search memes
        )

        print(f"Query {idx}: {test_image}")
        print(f"Saved query image as {query_resized}")
        if result_path:
            print(f"Top 4 similar images saved as {result_path}\n")
        else:
            print("No results found.\n")


def query_single_image(image_path, label_filter=1):
    normalized_name = image_path.replace("/", "_").replace("\\", "_")
    name, _ = os.path.splitext(normalized_name)
    query_image_path = f"./results/{name}_query.jpg"
    top_image_path = f"./results/{name}_top_result.jpg"

    # Save the original query image
    query_image = Image.open(image_path)
    if query_image.mode not in ["RGB"]:
        query_image = query_image.convert("RGB")
    query_image.save(query_image_path)

    query_vector = extractor.extract_features(image_path)
    results = milvus_manager.search_similar(query_vector, top_k=1)

    filtered_results = [
        hit
        for hit in results[0]
        if label_filter is None or hit["entity"]["label"] == label_filter
    ]

    if not filtered_results:
        print("No results found for the query image.")
        return

    top_result = filtered_results[0]
    filename = top_result["entity"]["filename"]

    # Keep the original dimensions for the top result image
    top_image = Image.open(filename)
    if top_image.mode not in ["RGB"]:
        top_image = top_image.convert("RGB")
    top_image.save(top_image_path)

    print(f"Query image saved as {query_image_path}")
    print(f"Top result image saved as {top_image_path}\n")
