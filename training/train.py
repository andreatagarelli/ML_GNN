import time
import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from models.mgat import MGAT
from models.mgcn import MGCN
from utils.util import accuracy, fmeasure, rlap, prediction, \
    expand_index
from utils.input_data import load_data, encode_onehot
from training.optimization import EarlyStopping
from models.weight__constraints import weight_clipper
import dgl
import warnings
from training.sampler import CustomSampler
from utils.params import set_params, load_config, print_arguments


def get_model(param, features, num_classes, nol, cuda):
    if param.model == 'mlgat':
        model = MGAT(num_layers=param.num_layers, in_dim=features.shape[1], num_hidden=param.hidden,
                     num_classes=num_classes, heads=[param.n_heads] * param.num_layers, activation=F.elu,
                     n_el=nol, aggregation=param.heads_mode, drop=param.dropout,
                     attn_drop=param.attn_dropout, negative_slope=param.alpha,
                     trainable_fweights=not param.nt, fusion_weights=param.fusion_weights,
                     ff_layers=param.fc_layers)

    elif param.model == 'mlgcn':
        model = MGCN(in_dim=features.shape[1], num_hidden=param.hidden, num_classes=num_classes,
                     num_layers=param.num_layers, activation=F.relu, n_el=nol, dropout=param.dropout,
                     trainable_fweights=not param.nt, fusion_weights=param.fusion_weights,
                     ff_layers=param.fc_layers)

    else:
        raise ValueError('Unknown GNN model.')
    if cuda:
        model = model.cuda()

    return model


def load_input_data(param):

    t = load_data(dataset_name=param.dataset,
                  num_features=param.input_features,
                  root_dir=param.data,
                  features_distribution=param.feat_distribution, train_percentage=param.train_percentage,
                  test_percentage=param.test_percentage,
                  feat_var=param.feat_variability, standardize=param.standardize)

    return t


def get_batch_size(nol, entity_batch):
    return int(nol * entity_batch)


def compute_test(model, test_g, test_feat, idx_test, test_labels, cuda, average='micro'):
    model.eval()
    with torch.no_grad():
        out = model.inference(test_g, test_feat)
        out = out[idx_test]
        loss_test = F.nll_loss(out, test_labels)
        if cuda:
            out = out.cpu()
            tlab = test_labels.cpu()
        else:
            tlab = test_labels
        fmeasure_test = fmeasure(tlab, prediction(out), average=average)
        rlap_test = rlap(encode_onehot(tlab.numpy()), out.numpy())

        return fmeasure_test, rlap_test, loss_test.item()


def train(param, data):
    cuda = torch.cuda.is_available() and not param.gpu < 0
    random.seed(param.seed)
    np.random.seed(param.seed)
    torch.manual_seed(param.seed)
    dgl.random.seed(param.seed)

    if cuda:
        torch.cuda.set_device(param.gpu)
        print('Using GPU device {}'.format(torch.cuda.get_device_name(param.gpu)))
        torch.cuda.manual_seed(param.seed)

    # Caricamento dei dati
    g, features, labels, idx_train, idx_val, idx_test, d_info = data
    if features.is_sparse:
        features = features.to_dense()  # due to possible errors o
        # f some DGL modules with sparse features

    validation = param.early_stop
    if len(idx_val) == 0:
        validation = False
        warnings.warn('No validation instances found.', UserWarning)

    nol = np.prod(d_info[1])  # number of elementary layers
    num_classes = int(labels.max()) + 1
    n = d_info[0]  # number of entities
    n_train = len(idx_train)  # number of training entities
    entity_batch = param.batch_size  # batch size

    if cuda:
        features = features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        g = g.to(features.device)

    # Model and optimizer
    model = get_model(param, features, num_classes, nol, cuda)
    shuffle = param.batch_shuffle
    if param.batch_size >= n_train:
        full_batch = True
        n_steps = 1
        shuffle = False
    else:
        full_batch = False
        n_steps = int(n_train / entity_batch) if n_train % entity_batch == 0 else int(n_train / entity_batch) + 1

    # Semi-supervised transductive setting
    train_g = val_g = test_g = g
    train_feat = val_feat = test_feat = features
    train_labels = labels
    test_labels = labels[idx_test]
    val_labels = labels[idx_val]
    train_nids = expand_index(idx_train, nol, n=n, sort=False)

    batch_size = get_batch_size(nol, entity_batch)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(param.num_layers)  # change the sampler for subsampling
    shuffler = CustomSampler(train_nids, n_train=n_train, nol=nol, entity_batch_size=entity_batch,
                             shuffle=shuffle)
    dataloader = dgl.dataloading.NodeDataLoader(train_g, train_nids, block_sampler=sampler, batch_size=batch_size,
                                                shuffle=False, drop_last=False, device=train_g.device, sampler=shuffler)

    def load_subtensor(t_feat, t_labels, t_seeds, t_input_nodes, device, t_steps):
        batch_i = t_feat[t_input_nodes].to(device)
        if full_batch:
            batch_l = t_labels[t_seeds[:n_train]]
        elif t_steps == n_steps and (n_train % entity_batch != 0):
            batch_l = t_labels[t_seeds[:n_train % entity_batch]].to(device)
        else:
            batch_l = t_labels[t_seeds[:entity_batch]].to(device)
        return batch_i, batch_l

    optimizer = optim.Adam(model.parameters(), lr=param.lr, weight_decay=param.weight_decay)

    if param.early_stop:
        stopper = EarlyStopping(patience=param.patience, maximize=param.maximize,
                                model_name=f'{param.model}_{param.dataset}_{param.feat_distribution}',
                                model_dir=param.checkpoint)

    loss_val_list = []  # Validation loss list
    acc_val_list = []  # Validation accuracy list
    loss_train_list = []  # Training loss list
    acc_train_list = []  # Training accuracy list
    t_total = time.time()  # Total training time
    for epoch in range(param.epochs):
        t = time.time()
        model.train()
        loss_batch = []  # loss of each batch
        acc_batch = []  # accuracy of each batch
        for steps,  (input_nodes, output_nodes, blocks) in enumerate(dataloader):

            batch_inputs, batch_labels = load_subtensor(train_feat, train_labels,
                                                        output_nodes, input_nodes, g.device, steps+1)

            blocks = [block.int().to(g.device) for block in blocks]

            optimizer.zero_grad()
            batch_output = model(blocks, batch_inputs)
            loss_train = F.nll_loss(batch_output, batch_labels)
            loss_train.backward()
            optimizer.step()

            # Non negative fusion weights
            weight_clipper(model.layers[-1].state_dict()['fusion'])

            acc_train = accuracy(batch_labels, prediction(batch_output))
            loss_batch.append(loss_train.data.item())
            acc_batch.append(acc_train.data.item())
            print('Epoch {:05d} | Step {:05d} | Loss Train {:.4f} | Train Acc {:.4f} | time {:.4f}s'
                  ''.format(epoch+1, steps+1, loss_train.item(), acc_train.item(),
                            time.time() - t))

        tot_loss = np.array(loss_batch).mean()
        tot_acc = np.array(acc_batch).mean()
        loss_train_list.append(tot_loss)
        acc_train_list.append(tot_acc)
        if validation:
            model.eval()
            with torch.no_grad():
                output = model.inference(val_g, val_feat)

                output = output[idx_val]
                loss_val = F.nll_loss(output, val_labels)
                loss_val_list.append(loss_val)
                acc_val = accuracy(val_labels, prediction(output))
                acc_val_list.append(acc_val)

                print('Epoch {:05d} | Loss Val {:.4f} | Val Acc {:.4f}'.format(epoch+1, loss_val.data.item(), acc_val))
                if param.early_stop:
                    if args.maximize:
                        score = acc_val
                    else:
                        score = loss_val.data.item()
                    if stopper.step(score, model, epoch):
                        break

    print("Optimization finished")
    print("Total training time: {:.4f}s".format(time.time() - t_total))

    if param.early_stop:
        print('Loading the best model....')
        model.load_state_dict(torch.load(stopper.save_dir))

    average = 'micro'
    f1_test, mrr_test, loss_test = compute_test(model, test_g, test_feat, idx_test, test_labels, cuda, average='micro')
    print("Test set results:", "loss= {:.4f}".format(loss_test),
          "F1-{} = {:.4f} ".format(average, f1_test),
          "MRR = {:.4f} ".format(mrr_test))

    return f1_test, mrr_test


if __name__ == '__main__':
    args = set_params(load_config())
    print()
    print_arguments(args)
    print()

    input_data = load_input_data(args)

    train(args, input_data)







