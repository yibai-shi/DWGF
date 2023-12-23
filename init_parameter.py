from argparse import ArgumentParser


def init_model():
    parser = ArgumentParser()

    parser.add_argument("--data_dir", default='data', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")

    parser.add_argument("--save_results_path", type=str, default='outputs', help="The path to save results.")

    parser.add_argument("--pretrain_dir", default='pretrain_models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--train_dir", default='train_models', type=str,
                        help="The output directory where the final model is stored in.")

    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="The path or name for the pre-trained bert model.")

    parser.add_argument("--tokenizer", default="bert-base-uncased", type=str,
                        help="The path or name for the tokenizer")

    parser.add_argument("--max_seq_length", default=None, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")

    parser.add_argument("--warmup_proportion", default=0.1, type=float)

    parser.add_argument("--freeze_bert_parameters", action="store_true", help="Freeze the last parameters of BERT.")

    parser.add_argument("--save_model", action="store_true", help="Save trained model.")

    parser.add_argument("--pretrain", action="store_true", help="Pre-train the model with labeled data.")

    parser.add_argument("--dataset", default='banking', type=str, required=True,
                        help="The name of the dataset to train selected.")

    parser.add_argument("--known_cls_ratio", default=0.75, type=float, required=True,
                        help="The number of known classes.")

    parser.add_argument('--seed', type=int, default=0, help="Random seed for initialization.")

    parser.add_argument("--rtr_prob", default=0.25, type=float,
                        help="Probability for random token replacement")

    parser.add_argument("--labeled_ratio", default=0.1, type=float,
                        help="The ratio of labeled samples in the training set.")

    parser.add_argument("--gpu_id", type=str, default='3', help="Select the GPU id.")

    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Batch size for training.")

    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")

    parser.add_argument("--pre_wait_patient", default=20, type=int,
                        help="Patient steps for pre-training Early Stop.")

    parser.add_argument("--num_pretrain_epochs", default=100, type=float,
                        help="The pre-training epochs.")

    parser.add_argument("--num_train_epochs", default=100, type=float,
                        help="The training epochs.")

    parser.add_argument("--lr_pre", default=5e-5, type=float,
                        help="The learning rate for pre-training.")

    parser.add_argument("--lr", default=1e-5, type=float,
                        help="The learning rate for training.")

    parser.add_argument("--grad_clip", default=1, type=float,
                    help="Value for gradient clipping.")

    parser.add_argument("--k", default=15, type=int, 
                        help='first-order neighborhood size for training')

    parser.add_argument("--alpha", default=0.3, type=float, 
                        help='loss weight for self-training')

    parser.add_argument('--tau', default=1, type=float, 
                        help='temperature for contrasitve learning')

    parser.add_argument('--interval', default=50, type=int,
                        help='interval for updating DWG')

    parser.add_argument('--r', default=2, type=int,
                        help='diffusion rounds')
    
    parser.add_argument('--t', default=2, type=int,
                        help='number of stacking filter layers')

    parser.add_argument("--g_k", default=15, type=int, 
                        help='first-order neighborhood size for inference')

    return parser
