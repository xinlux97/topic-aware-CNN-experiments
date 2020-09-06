import numpy as np

class VectorDict:
    def __init__(self):
        self.vector_dict = {}
        self.src_dict = {}
        self.src_dict_rev = {}
        self.vector_type = None
        self.embedding = None

    def reverse(self):
        for key in self.src_dict:
            self.src_dict_rev[self.src_dict[key]] = key

    def add_vector(self):
        if self.vector_type == 'glove':
            self.vector_dict['<pad>'] = self.vector_dict['pad']
            self.vector_dict['<unk>'] = self.vector_dict['unk']
            np.random.seed(10)
            self.vector_dict['</s>'] = np.random.random([self.vector_dict['you'].shape[0]]).astype(self.vector_dict['you'].dtype)
            np.random.seed(20)
            self.vector_dict['<Lua heritage>'] = np.random.random([self.vector_dict['you'].shape[0]]).astype(self.vector_dict['you'].dtype)
        elif self.vector_type == 'word2vec':
            np.random.seed(10)
            self.vector_dict.add('<pad>', np.random.random([500]).astype(self.vector_dict.get_vector('you').dtype))
            np.random.seed(20)
            self.vector_dict.add('<unk>', np.random.random([500]).astype(self.vector_dict.get_vector('you').dtype))
            np.random.seed(30)
            self.vector_dict.add('</s>', np.random.random([500]).astype(self.vector_dict.get_vector('you').dtype))
            np.random.seed(30)
            self.vector_dict['<Lua heritage>'] = np.random.random([self.vector_dict['you'].shape[0]]).astype(self.vector_dict['you'].dtype)
    
    def get_embedding(self, indices, dim):
        if self.vector_type == 'glove':
            return self.get_embedding_glove(indices, dim)
        elif self.vector_type == 'word2vec':
            return self.get_embedding_word2vec(indices, dim)
    
    def get_embedding_glove(self, indices, dim):
        all_embedding = []
        for batch in indices:
            batch_embedding = []
            for index in batch:
                word = self.src_dict_rev[index]
                if word in self.vector_dict:
                    embedding = self.vector_dict[word]
                else:
                    embedding = self.vector_dict["<unk>"]
                batch_embedding.append(embedding)
            all_embedding.append(batch_embedding)
        result_embedding = np.zeros([indices.shape[0],indices.shape[1],dim])
        result_embedding[:,:,:embedding.shape[0]] = np.array(all_embedding).astype(np.double)
        return result_embedding
    
    def get_embedding_word2vec(self, indices, dim):
        all_embedding = []
        for batch in indices:
            batch_embedding = []
            for index in batch:
                word = self.src_dict_rev[index]
                if word in self.vector_dict:
                    embedding = self.vector_dict.get_vector(word)
                else:
                    embedding = self.vector_dict.get_vector("<unk>")
                batch_embedding.append(embedding)
            all_embedding.append(batch_embedding)
        result_embedding = np.zeros([indices.shape[0],indices.shape[1],dim])
        result_embedding[:,:,:embedding.shape[0]] = np.array(all_embedding).astype(np.double)
        return result_embedding

vector_dict = VectorDict()
#vector_dict = np.load('./glove.npy')