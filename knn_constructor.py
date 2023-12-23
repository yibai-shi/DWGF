from utils.util import *
from utils.adjacency import *


class KNNConstructor:

    def __init__(self, args, model, mode):
        set_seed(args.seed)
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.sim_matrix = None

    def dump_knn_features(self, data):

        if self.mode == 'training':
            dataloader = data.train_semi_dataloader
            desc = "semi_dataset knn constructing"
        else:
            dataloader = data.test_dataloader
            desc = "test_dataset knn constructing"

        self.model.eval()
        total_features = []
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc=desc):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            X = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.no_grad():
                features = self.model(X)
            if isinstance(features, tuple):
                features = features[0]
            total_features.append(features)
            total_labels = torch.cat((total_labels, label_ids))
        
        self.labels = total_labels
        self.features = torch.vstack(total_features)
        self.features = F.normalize(self.features, dim=1)
        self.sim_matrix = self.features @ self.features.t()
        self.label_mask = (self.labels.unsqueeze(0) == self.labels.unsqueeze(1)).float()
        self.label_mask *= (self.labels.unsqueeze(0) >= 0).float()   
        self.label_mask = self.label_mask.to('cpu')   

    def initialize_graph(self, args):

        import faiss
        self.features = self.features.cpu().numpy()
        _, dim = self.features.shape[0], self.features.shape[1]
        index = faiss.IndexFlatIP(dim)  
        index.add(self.features)  
        if self.mode == 'training':
            sims, self.nbrs = index.search(self.features, args.k + 1) 
        else:
            sims, self.nbrs = index.search(self.features, args.g_k + 1) 
        self.knns = [(np.array(nbr, dtype=np.int32), 1 - np.array(sim, dtype=np.float32)) for nbr, sim in zip(self.nbrs, sims)]

        from scipy.sparse import csr_matrix
        eps = 1e-5
        n = len(self.knns)
        row, col, data = [], [], []
        for row_i, knn in enumerate(self.knns):
            nbrs, dists = knn
            for nbr, dist in zip(nbrs, dists):
                assert -eps <= dist <= 1 + eps, "{}: {}".format(row_i, dist)
                w = dist
                if nbr == -1:
                    continue
                if row_i == nbr:
                    assert abs(dist) < eps
                    continue
                row.append(row_i)
                col.append(nbr)
                w = 1 - w
                data.append(w)
        assert len(row) == len(col) == len(data)
        adj = csr_matrix((data, (row, col)), shape=(n, n))
        adj = build_symmetric_adj(adj, self_loop=True)

        indices, values, shape = sparse_mx_to_indices_values(adj)
        self.adj = indices_values_to_sparse_tensor(indices, values, shape)

    def build_DWG(self, args, cumulative=True):
        
        self.initialize_graph(args)
        adj = self.adj.to_dense()
        adj_diff = adj.clone()          
        self.dwg = adj.clone()

        # conduct diffusion
        r = args.r
        for i in range(r):
            adj_diff = adj_diff @ adj
            self.dwg = self.dwg + (1**i) * adj_diff if cumulative else adj_diff
        
        # filtering by relaxed similarity threshold
        self.dwg[self.sim_matrix<=0.3] = 0

        # normalization
        tmp = torch.max(self.dwg, dim=1, keepdim=True)
        self.dwg = 1.1 * (self.dwg / tmp.values)

        # levearge labeled subset
        self.dwg += self.label_mask
        self.dwg = torch.clamp(self.dwg, 0, 1)

        # final dwg
        self.dwg = self.dwg.to(self.device)
        self.dwg = self.dwg.detach()
