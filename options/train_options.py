from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.is_train = True

    def initialize(self):
        super().initialize()
        self._parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # 基础训练参数
        self._parser.add_argument('--save_interval', type=int, default=5, help='interval for saving models')
        self._parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
        self._parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        self._parser.add_argument('--learning_rate', type=float, default=8e-5, help='learning rate')
        self._parser.add_argument('--n_epoch', type=int, default=100, help='training epochs')
        self._parser.add_argument('--val_freq', type=int, default=1, help='validation frequency')
        self._parser.add_argument('--T_max', type=int, default=50, help="cosine learning rate period (iteration)")
        self._parser.add_argument('--eta_min', type=int, default=0, help="mininum learning rate")

    def parse(self):
        opt = super().parse()
        return opt
