import argparse

parser = argparse.ArgumentParser()

# Environment
parser.add_argument('--seed', type=int, default=2333, help='Global seed')
parser.add_argument('--gpus', type=str, help='GPU used')

# Training and Testing
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers created by the Pytorch dataloader')
parser.add_argument('--batch_size', type=int, default=2, help='Number of samples in one mini-batch')
parser.add_argument('--n_epochs', type=int, default=80, help='Number of epoches for the training process')
parser.add_argument('--output_path', type=str, help='Output path of the log and checkpoints')
parser.add_argument('--test', action=argparse.BooleanOptionalAction, help='Whether to test')
parser.add_argument('--test_after_train', action=argparse.BooleanOptionalAction, help='Whether to test after train')
parser.add_argument('--checkpoint', type=str, help='Checkpoint path for resuming or testing')

# Optimizer and Scheduler
parser.add_argument('--scheduler', type=str, default='onecycle', help='Scheduler name')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='Percentage of the cycle when the learning rate is increasing in the OneCycleLR')
parser.add_argument('--anneal_strategy', type=str, default='cos', help='Specify the cosine or linear annealing strategy')
parser.add_argument('--final_div_factor', type=int, default=10**5, help='Ratio of the maximum learning rate over the minimum learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for ReduceLROnPlateau')
parser.add_argument('--factor', type=float, default=0.1, help='Reduce factor for ReduceLROnPlateau')

# Data
parser.add_argument('--sigma', type=int, default=3, help='Sigma for heatmap')
parser.add_argument('--crop_size', type=int, default=256, help='The size of the randomly cropped part of the images')
parser.add_argument('--out_size', type=int, default=256, help='The size of the output images')
parser.add_argument('--downsample', type=int, default=8, help='The ratio by which the density maps shrink')
parser.add_argument('--seq_len', type=int, default=8, help='Sequence length of temporal data')

# Model
parser.add_argument('--model', type=str, help='Model name')
parser.add_argument('--hid_dim', type=int, default=128, help='Hidden dimension')
parser.add_argument('--frame_interval', type=int, default=5, help='Interval between frames')

args = parser.parse_args()