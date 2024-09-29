import torch
from pinecone_text.sparse import SpladeEncoder
from FlagEmbedding import BGEM3FlagModel


class TiledeskSpladeEncoder:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.splade = SpladeEncoder(device=self.device)

    def encode_documents(self, contents):
        # Logica di encoding specifica per SpladeEncoder

        doc_sparse_vectors = self.splade.encode_documents(contents)
        return doc_sparse_vectors

    def encode_queries(self,query):
        return  self.splade.encode_queries(query)


class TiledeskBGEM3:
    def __init__(self):
        self.use_fp16_bool = True if torch.cuda.is_available() else False
        self.model = BGEM3FlagModel('BAAI/bge-m3',
                               use_fp16=self.use_fp16_bool
                               )

    def encode_documents(self, contents):


        output_1 = self.model.encode(contents, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        dd = output_1['lexical_weights']
        doc_sparse_vectors = [
            {
                'indices': [int(k) for k in dd.keys()],
                'values': [float(dd[k]) for k in dd.keys()]
            }
            for dd in dd
        ]

        #for d in doc_sparse_vectors:
        #    print(d)

        return doc_sparse_vectors

    def encode_queries(self, query):
        query_encode = self.model.encode([query], return_dense=False, return_sparse=True, return_colbert_vecs=False)
        dd = query_encode['lexical_weights']
        doc_sparse_vectors = [
            {
                'indices': [int(k) for k in dd.keys()],
                'values': [float(dd[k]) for k in dd.keys()]
            }
            for dd in dd
        ]
        #print(doc_sparse_vectors[0])
        return doc_sparse_vectors[0]



class TiledeskSparseEncoders:
    def __init__(self, model_name):
        self.encoder = self._get_encoder(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _get_encoder(self, model_name):
        if model_name == "splade":
            return TiledeskSpladeEncoder()
        elif model_name == "bge-m3":
            return TiledeskBGEM3()
        else:
            raise ValueError("Unsupported model_name: {}. Supported values are 'splade' and 'bge-m3'.".format(model_name))

    def encode_documents(self, contents):
        if self.encoder:
            return self.encoder.encode_documents(contents)
        else:
            raise ValueError("No encoder has been initialized.")

    def encode_queries(self, query):
        if self.encoder:
            return self.encoder.encode_queries(query)
        else:
            raise ValueError("No encoder has been initialized.")

