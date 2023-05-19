import argparse
import yaml


def load_config(path='config.yml'):
    with open(path, "r") as ymlfile:
        return yaml.load(ymlfile, Loader=yaml.SafeLoader)


def set_params(conf = None):
    parser = argparse.ArgumentParser()
    # Framework parameters
    parser.add_argument('--seed', type=int, default=42
                        , help='Random seed.')
    parser.add_argument('-m', '--model', type=str, default='mlgat',
                        help="Model: mlgat or mlgcn.")
    parser.add_argument('-d', '--dataset', type=str, default='koumbia_10',
                        help='Name of input multilayer network.')
    parser.add_argument('-a', '--data', type=str, default='../data/nets/',
                        help='Data root directory.')
    parser.add_argument('-g', '--gpu', type=int, default=0, help="GPU.")  # un bucket 'attaccato' ad AWS SageMaker

    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs.')
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Numbers of hidden layers (K)")
    parser.add_argument('--hidden', type=int, default=32, help='Number of hidden neurons (d).')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--attn-dropout', type=float, default=0.5,
                        help='Attention dropout rate (1 - keep probability) for attention based models.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (L2 loss).')
    parser.add_argument('--n-heads', type=int, default=2, help='Number of attentions heads (Q)')
    parser.add_argument('--heads-mode', type=str, default='avg', help='Concatenate (concat) or averaging (avg)'
                                                                      ' the multiple (Q) attention heads')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Alpha (LeakyRelu function).')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience value for early stopping')
    parser.add_argument('--early-stop', action='store_true',
                        help='Whether to use early stopping regularization technique.')

    parser.add_argument('--input-features', type=int, default=64,
                        help="Number of random input node attributes (f).")
    parser.add_argument('--feat-distribution', type=str, default='gaussian', help='Features distribution, '
                                                                                      'e.g. gaussian, uniform, '
                                                                                      'mixed, etc.')
    parser.add_argument('--nt', action='store_false', help="Whether to make cross-layer aggregation weights learnable during training. nt will be True if weights are learnable")
    parser.add_argument('--fusion_weights', type=list, default=None,
                        help="Initialization of layer-wise aggregation weights.")

    parser.add_argument('--feat-variability', type=str, default='fixed',
                        help="How to assign random features between layers, i.e. features "
                             "per entity ('fixed') or per node ('layer')")

    parser.add_argument('--train_percentage', help="Percentage of training nodes for each class.",
                        default=25, type=int)
    parser.add_argument('--test_percentage', help="Percentage of testing nodes for each class.",
                        default=50, type=int)
    parser.add_argument('--standardize', action='store_true',
                        help='Whether to standardize node attributes.')
    parser.add_argument('--fc-layers', type=int, default=1, help='Number of hidden layers '
                                                                 'of the feed-forward neural network classifier.')
    parser.add_argument('--batch-size', type=int, default=4096, help='Batch size')
    parser.add_argument('--batch-shuffle', action='store_true', help='Whether to shuffle the'
                                                                             ' training batch after each epoch')

    args, _ = parser.parse_known_args()
    if conf is not None:
        for k in conf.keys():
            v = conf[k]
            if isinstance(v, dict):
                for k1 in v.keys():
                    v1 = v[k1]
                    setattr(args, k1, str2None(v1))

            else:
                setattr(args, k, str2None(v))
    return args


def str2None(v):
    if isinstance(v,str):
        if not v: return None

        if v.lower()=='none': return None

    return v


def print_arguments(args):
    print('List of arguments:')
    if isinstance(args, dict):
        for i, (k, v) in enumerate(args.items()):
            print("{} : {}".format(k, v), end="; ")
            if (i + 1) % 7 == 0:
                print()

    else:
        for i, arg in enumerate(vars(args)):
            print("{} : {}".format(arg, getattr(args, arg)) , end="; ")
            if (i + 1) % 7 == 0:
                print()

