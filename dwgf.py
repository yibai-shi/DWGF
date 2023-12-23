from utils.util import *
from utils.memory import *
from dataloader import *
from model import *
from knn_constructor import *
from init_parameter import *
from pretrain import *


class ModelManager:

    def __init__(self, args, data, pretrained_model=None, initial_knn=None):

        set_seed(args.seed)
        self.pretrained_model = pretrained_model
        self.knn = initial_knn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForGraph(args.bert_model, data.num_labels)
        self.model.to(self.device)
        self.model_m = BertForGraph(args.bert_model, data.num_labels)
        self.model_m.to(self.device)
        self.load_pretrained_model()
        self.m = 0.8

        self.num_train_optimization_steps = int(len(data.train_semi_dataset) / args.train_batch_size) * args.num_train_epochs

        self.optimizer, self.scheduler = self.get_optimizer(args)

        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.generator = view_generator(self.tokenizer, args.rtr_prob, args.seed)

        self.memory_bank = MemoryBank(args, len(data.train_semi_dataset), 256)
        fill_memory_bank(data.train_semi_dataloader, self.model, self.memory_bank, self.generator, self.device)

    def load_pretrained_model(self):
        pretrained_dict = self.pretrained_model.state_dict()
        classifier_params = ['classifier.weight', 'classifier.bias']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)
        self.model_m.load_state_dict(pretrained_dict, strict=False)
        for _, param in self.model_m.named_parameters():
            param.requires_grad = False

    def get_optimizer(self, args):
        num_warmup_steps = int(args.warmup_proportion * self.num_train_optimization_steps)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=self.num_train_optimization_steps)
        return optimizer, scheduler

    def momentum_update_encoder_m(self):
        for param_q, param_m in zip(self.model.parameters(), self.model_m.parameters()):
            param_m.data = param_m.data * self.m + param_q.data * (1. - self.m)

    def get_features_labels(self, dataloader, model, args):

        model.eval()
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="stage2-feature extracting"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            X = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.no_grad():
                feature, _, _ = model(X)

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels

    def update(self, epoch, args, data):
        if (epoch + 1) % args.interval == 0:
            self.knn = KNNConstructor(args, self.model, mode='training')
            self.knn.dump_knn_features(data)
            self.knn.build_DWG(args)

    def D_contrast(self, x, args, batch_id, dwg):
        x_dis = F.normalize(x, dim=1) @ F.normalize(self.memory_bank.features.t(), dim=0)
        x_dis = torch.exp(args.tau * x_dis)
        x_dis_sum = torch.sum(x_dis, 1)
        # SequentialSampler -> 'dwg[batch_id * args.train_batch_size : batch_id * args.train_batch_size + len(x_dis)]' means current mini-batch data
        x_dis_sum_pos = torch.sum(x_dis * dwg[batch_id * args.train_batch_size : batch_id * args.train_batch_size + len(x_dis)], 1)
        loss = -torch.log(x_dis_sum_pos * (x_dis_sum ** (-1))).mean()
        return loss
    
    def train(self, args, data):
        for epoch in range(int(args.num_train_epochs)):
            print('---------------------------')
            print(f'training epoch:{epoch}')

            self.memory_bank.reset()

            l_contrast = 0
            l_self = 0

            if epoch == 0:
                feats, _ = self.get_features_labels(data.train_semi_dataloader, self.model, args)
                feats = feats.cpu().numpy()
                km = KMeans(n_clusters=data.num_labels).fit(feats)
                self.model.cluster_layer.data = torch.tensor(km.cluster_centers_).to(self.device)

            self.model.train()

            for batch_id, batch in enumerate(tqdm(data.train_semi_dataloader, desc='stage2-training')):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                X = {"input_ids": self.generator.random_token_replace(input_ids.cpu()).to(self.device), 
                     "attention_mask": input_mask, "token_type_ids": segment_ids}
                _, _, features_m = self.model_m(X)
                _, q, features = self.model(X)

                self.memory_bank.update(features_m, label_ids)

                nContrast_loss = self.D_contrast(features, args, batch_id, self.knn.dwg)
                l_contrast += nContrast_loss

                self_training_loss = self.model.loss_self(q)
                l_self += self_training_loss

                loss = nContrast_loss + args.alpha * self_training_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)

                self.optimizer.step()
                self.scheduler.step()
                self.momentum_update_encoder_m()
                self.optimizer.zero_grad()

            self.update(epoch, args, data)

        if args.save_model:
            self.save_model(args)

    def laplacian_filtering(self, A, X, t):
        A[A > 0] = 1
        A = A.cpu().numpy()
        adj_ = sp.coo_matrix(A)

        ident = sp.eye(A.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

        laplacian = ident - adj_normalized
        ident = torch.eye(A.shape[0])
        indices, values, shape = sparse_mx_to_indices_values(laplacian)
        laplacian = indices_values_to_sparse_tensor(indices, values, shape)
        laplacian = laplacian.to_dense()
        
        for _ in range(t):
            X = (ident - 0.5 * laplacian) @ X

        return X

    def evaluation(self, args, data, mode):

        feats, labels = self.get_features_labels(data.test_dataloader, self.model, args)
        feats = feats.cpu().numpy()

        if mode == 'gsf':
            knn = KNNConstructor(args, self.model, mode='inference')
            knn.dump_knn_features(data)
            knn.initialize_graph(args)
            knn.adj = knn.adj.to_dense()
            feats = self.laplacian_filtering(knn.adj, feats, args.t)
   
        km = KMeans(n_clusters=data.num_labels, n_init=20).fit(feats)

        y_pred = km.labels_
        y_true = labels.cpu().numpy()

        results = clustering_score(y_true, y_pred)
        print('results', results)

        score = metrics.silhouette_score(feats, km.labels_)
        print(f'silhouette_score:{score}')

        self.test_results = results

        self.save_results(args)

    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.seed, args.known_cls_ratio, args.labeled_ratio, args.k, args.r, args.alpha, args.tau, args.g_k, args.t]
        names = ['dataset', 'seed', 'known_cls_ratio', 'labeled_ratio', 'k', 'r', 'alpha', 'tau', 'g_k', 't']
        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(self.test_results, **vars_dict)
        keys = list(results.keys())
        values = list(results.values())

        file_name = 'results.csv'
        results_path = os.path.join(args.save_results_path, file_name)

        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results, index=[1])
            df1 = df1.append(new, ignore_index=True)
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)

        print('test_results', data_diagram)
    
    def save_model(self, args):
        if not os.path.exists(args.pretrain_dir):
            os.makedirs(args.pretrain_dir)
        model_file = os.path.join(args.pretrain_dir, 'step2_{}.pth'.format(args.dataset))
        torch.save(self.model.state_dict(), model_file)

    def load_model(self, args):
        model_file = os.path.join(args.pretrain_dir, 'step2_{}.pth'.format(args.dataset))
        self.model.load_state_dict(torch.load(model_file))


if __name__ == '__main__':

    parser = init_model()
    args = parser.parse_args()
    data = Data(args)

    manager_p = PretrainModelManager(args, data)
    manager_p.train(args, data)
    # manager_p.load_model(args)
    manager_p.evaluation(args, data)

    knn_constructor = KNNConstructor(args, manager_p.model, mode='training')
    knn_constructor.dump_knn_features(data)
    knn_constructor.build_DWG(args)

    manager_2 = ModelManager(args, data, manager_p.model, knn_constructor)
    manager_2.train(args, data)
    # manager_2.load_model(args)
    manager_2.evaluation(args, data, mode='org')
    manager_2.evaluation(args, data, mode='gsf')
