from pymilvus import MilvusClient


class MilvusManager:
    def __init__(
        self, uri, collection_name, dimension=1280, metric_type="COSINE"
    ):
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name

        # Create or reset collection
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vector_field_name="vector",
            dimension=dimension,
            auto_id=True,
            enable_dynamic_field=True,
            metric_type=metric_type,
        )

    def insert_image(self, vector, metadata):
        self.client.insert(
            self.collection_name, {"vector": vector, **metadata}
        )

    def search_similar(self, query_vector, top_k=1):
        return self.client.search(
            self.collection_name,
            data=[query_vector],
            output_fields=["filename"],
            search_params={"metric_type": "COSINE"},
            limit=top_k,
        )
