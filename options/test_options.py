from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--checkpoints_dir', type=str, default='./test_checkpoints', help='models are saved here')
        self._parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        self._parser.add_argument('--test_file_name', type=str, default='results/results.txt',
                                  help='txt path to save results')
        self._parser.add_argument('--n_ensemble', type=int, default=20,
                                  help='crop method for test: five points crop or nine points crop or random crop for several times')
        self._parser.add_argument('--flip', type=bool, default=False, help='if flip images when testing')
        self._parser.add_argument('--resize', type=bool, default=False, help='if resize images when testing')
        self._parser.add_argument('--size', type=int, default=224, help='the resize shape')
        self.is_train = False
