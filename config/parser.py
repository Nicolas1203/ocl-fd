import argparse
import yaml

class Parser:
    """
    Command line parser based on argparse. This also includes arguments sanity check.
    """

    def __init__(self):
        parser = argparse.ArgumentParser(description="Pytorch UCL, with tensorboard logs.")

        # Configuration parameters
        parser.add_argument('--config', default=None, help="Path to the configuration file for the training to launch.")
        # Training parameters
        parser.add_argument('--train', dest='train', action='store_true')
        parser.add_argument('--test', dest='train', action="store_false")
        parser.add_argument('--epochs', default=1, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('-b', '--batch-size', default=10, type=int,
                            help='mini-batch size (default: 10)')
        parser.add_argument('--learning-rate', '-lr', default=0.1, type=float, help='Initial learning rate')
        parser.add_argument('--momentum', default=0, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                            metavar='W', help='weight decay (default: 0)')
        parser.add_argument('--optim', default='SGD', choices=['Adam', 'SGD'])
        parser.add_argument("--save-ckpt", action='store_true', help="whether to save chekpoints or not")
        parser.add_argument('--seed', type=int, default=0, help='Random seed to use.')
        # Logs parameters
        parser.add_argument('--tag', '-t', default='', help="Base name for graphs and checkpoints")
        parser.add_argument('--tb-root', default='./runs/', help="Where do you want tensorboards graphs ?")
        parser.add_argument('--logs-root', default='./logs', help="Defalt root folder for writing logs.")
        parser.add_argument('--results-root', default='./results/', help='Where you want to save the results ?')
        parser.add_argument('--tensorboard', action='store_true')
        parser.add_argument('--verbose', action='store_true')
        # checkpoints params
        parser.add_argument('--ckpt-root', default='./checkpoints/', help='Directory where to save the model.')
        parser.add_argument('--resume', '-r', action='store_true', 
                            help="Resume old training. Setup model state and buffer state.")
        parser.add_argument('--model-state')
        parser.add_argument('--buffer-state')
        ##########
        # MODELS #
        ##########
        parser.add_argument('--head', default='mlp')
        # Resnet parameters
        parser.add_argument('--proj-dim', type=int, default=128)
        parser.add_argument('--nb-channels', type=int, default=3, 
                            help="Number of channels for the input image.")
        parser.add_argument('--eval-proj', action='store_true', help="Use projection for inference. (default is representation.)")
        parser.add_argument('--pretrained', action='store_true', help="Use a pretrained model if available.")
        # Intermediate classifier
        parser.add_argument('--supervised', action='store_true', help="Pseudo labels or true labels ?")
        parser.add_argument('--dim-int', type=int, default=512)
        parser.add_argument('-nf', type=int, default=64, help="Number of feature for Resnet18. Set nf=20 for reduced resnet18, nf=64 for full.")
        #####################
        # Dataset parameters
        #####################
        parser.add_argument('--data-root-dir', default='/data/',
                            help='Root dir containing the dataset to train on.')
        parser.add_argument('--min-crop', type=float, default=0.2, help="Minimum size for cropping in data augmentation. range (0-1)")
        parser.add_argument('--dataset', '-d', default="cifar10",
                            choices=['mnist', 'fmnist', 'cifar10', 'cifar100', 'tiny', 'sub', 'yt'],
                            help='Dataset to train on')
        parser.add_argument('--training-type', default='inc', choices=['uni', 'inc', 'blurry'],
                            help='How to feed the data to the network (incremental context or not)')
        parser.add_argument('--n-classes', type=int, default=10,
                            help="Number of classes in database.")
        parser.add_argument("--img-size", type=int, default=32, help="Size of the square input image")
        parser.add_argument('--num-workers', '-w', type=int, default=0, help='Number of workers to use for dataloader.')
        parser.add_argument("--n-tasks", type=int, default=5, help="How many tasks do you want ?")
        parser.add_argument("--labels-order", type=int, nargs='+', help="In which order to you want to see the labels ? Random if not specified.")
        parser.add_argument("--blurry-scale", type=int, default=1500)
        # Contrastive loss parameters
        parser.add_argument('--temperature', default=0.07, type=float, 
                            metavar='T', help='temperature parameter for softmax')
        # Memory parameters
        parser.add_argument('--mem-size', type=int, default=200, help='Memory size for continual learning')  # used also for ltm
        parser.add_argument('--mem-batch-size', '-mbs', type=int, default=200, help="How many images do you want to retrieve from the memory/ltm")  # used also for ltm
        parser.add_argument('--buffer', default='reservoir', help="What buffer do you want ? See available buffers in utils/name_match.py")
        parser.add_argument('--drop-method', default='random', choices=['random'], help="How to drop images from memory when adding new ones.")
        parser.add_argument('--mem-iters', default=1, type=int, help="Number of iterations on memory")
        # Learner parameter
        parser.add_argument('--learner', help='What learner do you want ? See list of available learners in utils/name_match.py')
        parser.add_argument('--debug', action='store_true')
        # Inference parameters
        parser.add_argument('--eval-mem', action='store_true', dest='eval_mem')
        parser.add_argument('--eval-random', action='store_false', dest='eval_mem')
        parser.add_argument('--lab-pc', type=int, default=20, help="Number of labeled images per class to use in unsupervised evaluation.")
        # Multi runs arguments
        parser.add_argument('--n-runs', type=int, default=1, help="Number of runs, with different seeds each time.")
        parser.add_argument('--start-seed', type=int, default=0, help="First seed to use.")
        parser.add_argument('--run-id', type=int, help="Id of the current run in multi run.")
        parser.add_argument('--kornia', action='store_true', dest='kornia')
        parser.add_argument('--no-kornia', action='store_false', dest='kornia')
        # Aug params
        parser.add_argument('--n-augs', type=int, default=1)
        parser.add_argument('--no-aug-trick', action='store_true', help="If true, FD will train on B + \sum_Aug(B). If False, FD is trained on \sum_Aug(B) only.")
        parser.add_argument('--tf-type', default="full", choices=['full', 'partial'], help=' Data augmentation sequence to use.')
        # Projection trick
        parser.add_argument('--drop-fc', action="store_true", help="Drop the final FC layer and use memory data for transfer learning.")
        # FD arguments
        parser.add_argument('--var', type=float, default=0.14, help="Variance for fd loss. kappa=1/var for VMF and kappa=1/2*var for AGD.")
        parser.add_argument('--fd-loss', help="vmf or agd loss", default='vmf')
        parser.add_argument('-mu', type=float, default=1.0, help="mu value for FD model.")
        parser.add_argument('--norm-all-classes', action='store_true', help="Use this to normalize using every classes known.")
        # DERpp arguments
        parser.add_argument('--derpp-alpha', type=float, default=0.1, help="Values of alpha un der++ loss")
        parser.add_argument('--derpp-beta', type=float, default=0.5, help="Values of beta un der++ loss")
        # WandB argument
        parser.add_argument("--no-wandb", action='store_true')
        parser.add_argument('--wandb-watch', action='store_true', help="Watch the models gradient and parameters into wandb (can be slow).")
        parser.add_argument("--sweep", action='store_true', help="Run the code with sweep for HP search.")
        parser.set_defaults(train=True, eval_mem=True, kornia=True, daa=True, fixed_means=True)
        self.parser = parser

    def parse(self, arguments=None):
        if arguments is not None:
            self.args = self.parser.parse_args(arguments)
        else:
            self.args = self.parser.parse_args()
        self.load_config()
        self.check_args()
        return self.args

    def load_config(self):
        if self.args.config is not None:
            with open(self.args.config, 'r') as f:
                cfg = yaml.safe_load(f)
                for key in cfg:
                    setattr(self.args, key, cfg[key])
            f.close()

    def check_args(self):
        """Modify default arguments values depending on the method and dataset.
        """
        #############################
        # Dataset parameters sanity #
        #############################
        if self.args.dataset == 'cifar10':
            self.args.img_size = 32
            self.args.n_classes = 10
            self.args.dim_in = 8 * self.args.nf
        if self.args.dataset == 'cifar100': 
            self.args.img_size = 32
            self.args.n_classes = 100
            self.args.dim_in = 8 * self.args.nf
        if self.args.dataset == 'tiny':
            self.args.img_size = 64
            self.args.n_classes = 200
            self.args.dim_in = 32 * self.args.nf            
            # Style parameters specific to tinyIN (TODO: add this to configs)
            self.min_style_alpha=0.4
            self.max_style_alpha=0.8
        if self.args.dataset == 'sub':
            self.args.img_size = 224
            self.args.n_classes = 10
            self.args.dim_in = 392 * self.args.nf
        if self.args.dataset == 'yt':
            self.args.img_size = 224
            self.args.n_classes = 2
            self.args.dim_in = 392 * self.args.nf

        ##############################
        # Learners parameters sanity #
        ##############################
        if self.args.learner == 'STAM':
            self.args.batch_size = 1

