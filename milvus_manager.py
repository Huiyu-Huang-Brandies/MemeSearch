from pymilvus import MilvusClient


class MilvusManager:
    def __init__(
        self,
        uri,
        collection_name,
        dimension=1280,
        metric_type="COSINE",
        reset_collection=False,
    ):
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name

        if reset_collection and self.client.has_collection(
            collection_name=self.collection_name
        ):
            self.client.drop_collection(collection_name=self.collection_name)
            print(f"Collection '{self.collection_name}' has been reset.")

        if not self.client.has_collection(
            collection_name=self.collection_name
        ):
            self.client.create_collection(
                collection_name=self.collection_name,
                vector_field_name="vector",
                dimension=dimension,
                auto_id=True,
                enable_dynamic_field=True,
                metric_type=metric_type,
            )
            print(f"Collection '{self.collection_name}' created.")

    def insert_image(self, vector, metadata):
        self.client.insert(
            self.collection_name, {"vector": vector, **metadata}
        )

    def search_similar(self, query_vector, top_k=1):
        return self.client.search(
            self.collection_name,
            data=[query_vector],
            output_fields=["filename", "label"],
            search_params={"metric_type": "COSINE"},
            limit=top_k,
        )

    def is_image_in_database(self, filename):
        results = self.client.query(
            self.collection_name,
            expr=f'filename == "{filename}"',
            output_fields=["filename"],
        )
        return len(results) > 0

    def update_image_metadata(self, filename, metadata):
        """
        Update metadata for an existing image in the database.
        """
        self.client.update(
            self.collection_name,
            expr=f'filename == "{filename}"',
            set_fields=metadata,
        )
        print(f"Updated metadata for {filename}: {metadata}")

    def get_image_metadata(self, filename):
        try:
            results = self.client.query(
                collection_name=self.collection_name,
                expr=f'filename == "{filename}"',
                output_fields=["filename", "label", "category"],
            )
            return results[0] if results else None
        except Exception as e:
            print(f"Failed to query metadata for {filename}: {e}")
            return None

    def create_collection(self, dimension=1280, metric_type="COSINE"):
        if not self.client.has_collection(
            collection_name=self.collection_name
        ):
            self.client.create_collection(
                collection_name=self.collection_name,
                vector_field_name="vector",
                dimension=dimension,
                auto_id=True,
                enable_dynamic_field=True,
                metric_type=metric_type,
            )
            print(f"Collection '{self.collection_name}' created.")
        else:
            print(f"Collection '{self.collection_name}' already exists.")

    def clear_collection(self):
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
            print(f"Collection '{self.collection_name}' cleared.")
        else:
            print(f"Collection '{self.collection_name}' does not exist.")
