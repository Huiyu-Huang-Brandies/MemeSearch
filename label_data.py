import os
from feature_extractor import FeatureExtractor
from milvus_manager import MilvusManager

# Initialize components
extractor = FeatureExtractor("efficientnet_b0")
milvus_manager = MilvusManager(
    uri="example.db", collection_name="meme_embeddings"
)

# Paths to data folders
meme_folder = "./data/train_memes"
non_meme_folder = "./data/train_non_memes"


def process_images(folder, label=None, is_non_meme=False):
    """
    Process images in a folder, checking and updating labels as needed.
    - If the image exists but lacks metadata, update it.
    - If the image doesn't exist, insert it.
    """
    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.endswith((".png", ".jpg", ".jpeg", ".img")):
                filepath = os.path.join(root, filename)
                metadata = {
                    "filename": filepath,
                    "label": 0 if is_non_meme else 1,
                    "category": (
                        os.path.basename(root) if is_non_meme else None
                    ),
                }

                # # Check if the image already exists in the database
                # existing_metadata = milvus_manager.get_image_metadata(filepath)
                # if existing_metadata:
                #     # Update metadata if label or category is missing
                #     if (
                #         "label" not in existing_metadata
                #         or existing_metadata["label"] is None
                #         or (
                #             is_non_meme and "category" not in existing_metadata
                #         )
                #     ):
                #         milvus_manager.update_image_metadata(
                #             filepath, metadata
                #         )
                #         print(f"Updated {filename} with metadata: {metadata}")
                #     else:
                #         print(f"Skipping {filename} (already labeled)")
                # else:
                #     # Insert new image if it doesn't exist
                image_embedding = extractor.extract_features(filepath)
                milvus_manager.insert_image(image_embedding, metadata)


if __name__ == "__main__":
    try:
        milvus_manager.clear_collection()
        milvus_manager.create_collection()

        print("Processing non-meme images...")
        process_images(non_meme_folder, is_non_meme=True)

        # Process meme images
        print("Processing meme images...")
        process_images(meme_folder, label=1)
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        # Clean up resources
        print("Shutting down gracefully...")
