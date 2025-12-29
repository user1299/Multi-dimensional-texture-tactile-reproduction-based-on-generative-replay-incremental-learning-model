def add_parser(parser):
    parser.add_argument('--dataset_path', type=str, default="/media/ubuntu/DATA/dy/data/hatt_small/", help='dataset path')
    parser.add_argument('--images_dir', type=str, default="/media/ubuntu/DATA/dy/data/image/", help='dataset path')
    parser.add_argument('--d_model', type=int, default=64, help='Sequence Elements embedding dimension')
    parser.add_argument('--d_ff', type=int, default=128, help='Second Embedded representation')
    parser.add_argument('--bi_dir', type=int, default=1, help='use bidirectional Mamba?')
    parser.add_argument('--d_state', type=int, default=16, help='d_state parameter of Mamba')#1-32   ธ฿
    parser.add_argument('--d_conv', type=int, default=4, help='d_conv parameter of Mamba')#1-4
    parser.add_argument('--e_fact', type=int, default=1, help='expand factor parameter of Mamba')

    parser.add_argument('--e_layers', type=int, default=1, help='layers of encoder')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')#0.2-0.4, step=0.05   ตอ
    parser.add_argument('--activation', type=str, default='gelu', help='gelu,relu')
    parser.add_argument('--use_cpu', default=False, action='store_true', help='use gpu')
    parser.add_argument('--gpu', type=int, default=1, help='gpu')
    parser.add_argument('--use_multi_gpu', default=False, action='store_true', help='use multiple gpus')
    parser.add_argument('--device_ids', type=str, default='0,1', help='logic gpu ids, but assigned')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use for training')

    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    # basic config
    parser.add_argument('--seq_len', type=int, default=200, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=100, help='start token length')
    parser.add_argument('--pred_len', type=int, default=200, help='prediction sequence length')

    parser.add_argument('--task', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='accel', help='target feature in S or MS task')
    parser.add_argument('--dataset_type',       type=str,   default='custom',       help='dataset type')
    parser.add_argument('--freq',               type=str,   default='h',
        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--embed',              type=str,   default='timeF',        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--num_workers',        type=int,   default=4,              help='data loader num workers')

    # model saving path
    parser.add_argument('--checkpoints',        type=str,   default='/media/ubuntu/DATA/dy/TCLN/checkpoints',    help='location of model checkpoints')
    parser.add_argument('--results',            type=str,   default='/media/ubuntu/DATA/dy/TCLN/results',        help='location of model results')
    parser.add_argument('--predictions',        type=str,   default='/media/ubuntu/DATA/dy/TCLN/predictions',    help='location of model predictions')
    parser.add_argument('--datasave',           type=str, default='/media/ubuntu/DATA/dy/TCLN/datasave',         help='data save')

    # dataset channels
    parser.add_argument('--enc_in',             type=int,   default=4,             help='encoder input size')
    parser.add_argument('--dec_in',             type=int,   default=4,             help='decoder input size')
    parser.add_argument('--c_out',              type=int,   default=4,             help='output size')
    parser.add_argument('--timestamp_dim',      type=int,   default=1,              help='how many values does each timestamp record?')
    parser.add_argument('--relation',           type=str,   default='spearman',     help='algorithm for relationship')
    parser.add_argument('--itr',                type=int,   default=1,              help='experiments times')
    parser.add_argument('--train_epochs',       type=int,   default=10000,            help='train epochs')
    parser.add_argument('--batch_size',         type=int,   default=32,             help='batch size of train input data')
    parser.add_argument('--patience',           type=int,   default=8,             help='early stopping patience')
    parser.add_argument('--learning_rate',      type=float, default=0.0008,        help='optimizer learning rate')#0.0001-0.0016
    parser.add_argument('--loss',               type=str,   default='mse',          help='loss function, choose [mse, rmse, mae, mape, huber]')
    parser.add_argument('--lradj',              type=str,   default='type3',        help='adjust learning rate')
    parser.add_argument('--pct_start',          type=float, default=0.3,            help='how many epochs should the lr_rate get to max?')
    parser.add_argument('--opt',                type=str,   default='adam',         help='optimizer chosen from [Adam, SGD]')
    parser.add_argument('--revin',              type=int,   default=1,              help='RevIN; True 1 False 0')
    parser.add_argument('--affine',             type=int,   default=0,              help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last',      type=int,   default=0,              help='0: subtract mean; 1: subtract last')
    parser.add_argument('--embed_type',         type=int,   default=3,              help='0: value \
                                                                                      1: value + positional \
                                                                                      2: value + temporal \
                                                                                      3: value + positional + temporal')
    parser.add_argument('--SRA',                            default=False,  action='store_true',
                        help='use series-relation-aware decider?')
    parser.add_argument('--ch_ind',             type=int,   default=1,              help='forced channel independent?')
    parser.add_argument('--stride',             type=int,   default=4,              help='stride, half or full length of patch_len')
    parser.add_argument('--patch_len',          type=int,   default=8,             help='patch length')
    parser.add_argument('--pos_embed_type',     type=str,   default='sincos',       help='how do you generate positional encoding for Encoder')
    parser.add_argument('--pos_learnable',                  default=True,  action='store_true',
                        help='use fixed or learned Position Encoding')
    parser.add_argument('--residual',       type=int,       default=1,          help='residual connection?')
    parser.add_argument('--deform_patch', default=False, action="store_true", help='deform_patch')




