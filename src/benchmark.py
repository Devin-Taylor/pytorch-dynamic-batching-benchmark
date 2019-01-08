'''
This code presents a benchmark algorithm for dynamic batching implemented in
PyTorch using the SPINN model and SNLI dataset.

NOTE aspects of this code were adapted from https://github.com/nearai/torchfold
@misc{illia_polosukhin_2018_1299387,
  author       = {Illia Polosukhin and
                  Maksym Zavershynskyi},
  title        = {nearai/torchfold: v0.1.0},
  month        = jun,
  year         = 2018,
  doi          = {10.5281/zenodo.1299387},
  url          = {https://doi.org/10.5281/zenodo.1299387}
}

'''

__author__ = "Devin Taylor"
__version__ = "1.0"
__maintainer__ = "Devin Taylor"
__email__ = "dev.t03@gmail.com"

import argparse
import json
import os
import sys
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchtext import data, datasets

import torchfold

ROOT = "../data"

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='SPINN')
parser.add_argument('--dynamic', default=False ,type=str2bool, nargs='?')
parser.add_argument('--gpu', default=False ,type=str2bool, nargs='?')
args, _ = parser.parse_known_args(sys.argv)
args.cuda = args.gpu and torch.cuda.is_available()

results = {
    "epoch": [],
    "batch": [],
    "time": [],
    "sample": [],
    "treesize": [],
    "batch_size": []
}

class TreeNode(object):
    def __init__(self, leaf=None, left=None, right=None):
        self.id = leaf
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.id is not None

class Tree(object):
    def __init__(self, label, premise_transitions, premise, inputs_vocab, answer_vocab):
        self.label = answer_vocab.stoi[label] - 1
        queue = []
        idx, transition_idx = 0, 0
        while transition_idx < len(premise_transitions):
            t = premise_transitions[transition_idx]
            transition_idx += 1
            if t == 'shift':
                queue.append(TreeNode(leaf=inputs_vocab.stoi[premise[idx]]))
                idx += 1
            else:
                n_left = queue.pop()
                n_right = queue.pop()
                queue.append(TreeNode(left=n_left, right=n_right))
        assert len(queue) == 1
        self.root = queue[0]

class TreeLSTM(nn.Module):
    '''
    Implementation of the Tree-LSTM model:
    https://arxiv.org/pdf/1503.00075.pdf
    '''
    def __init__(self, num_units):
        super(TreeLSTM, self).__init__()
        self.left = nn.Linear(num_units, 5 * num_units)
        self.right = nn.Linear(num_units, 5 * num_units)

    def forward(self, left_in, right_in):
        lstm_in = self.left(left_in[0])
        lstm_in += self.right(right_in[0])
        a, i, f1, f2, o = lstm_in.chunk(5, 1)
        c = (a.tanh() * i.sigmoid() + f1.sigmoid() * left_in[1] +
             f2.sigmoid() * right_in[1])
        h = o.sigmoid() * c.tanh()
        return h, c

class SPINN(nn.Module):
    '''
    Implementation of the SPINN model:
    https://arxiv.org/pdf/1603.06021.pdf
    '''
    def __init__(self, n_classes, size, n_words):
        super(SPINN, self).__init__()

        self._device = torch.device("cpu")

        self.size = size
        self.tree_lstm = TreeLSTM(size)
        self.embeddings = nn.Embedding(n_words, size)
        self.out = nn.Linear(size, n_classes)

    def to(self, device):
        '''
        Move model to device specified by parameter
        '''
        self._device = device
        self.tree_lstm.to(device)
        self.embeddings.to(device)
        self.out.to(device)
        return self

    def leaf(self, word_id):
        embedded = self.embeddings(word_id.to(self._device))
        return embedded, torch.zeros((word_id.size(0), self.size)).type(dtype=torch.float32)

    def branches(self, left_h, left_c, right_h, right_c):
        return self.tree_lstm((left_h.to(self._device), left_c.to(self._device)), (right_h.to(self._device), right_c.to(self._device)))

    def logits(self, encoding):
        return self.out(encoding)

def encode_tree_regular(model, tree):
    def encode_node(node):
        if node.is_leaf():
            test = np.asarray([node.id], dtype=int)
            x = torch.from_numpy(test)
            return model.leaf(x)
        else:
            left_h, left_c = encode_node(node.left)
            right_h, right_c = encode_node(node.right)
            return model.branches(left_h, left_c, right_h, right_c)
    encoding, _ = encode_node(tree.root)
    return model.logits(encoding)

def encode_tree_fold(fold, tree):
    def encode_node(node):
        if node.is_leaf():
            return fold.add('leaf', node.id).split(2) # split by 2 because binary trees
        else:
            left_h, left_c = encode_node(node.left)
            right_h, right_c = encode_node(node.right)
            return fold.add('branches', left_h, left_c, right_h, right_c).split(2)
    encoding, _ = encode_node(tree.root)
    return fold.add('logits', encoding)

def tree_size(root: TreeNode):
    '''
    Count the number of nodes in a tree
    '''
    if root is None:
        return 0
    return 1 + tree_size(root.right) + tree_size(root.left)

class ShiftReduceField(data.Field):
    '''
    Modified from torchtext v0.2.3 as this class
    has been depricated in torchtext v0.4
    '''
    def __init__(self):
        super(ShiftReduceField, self).__init__(preprocessing=lambda parse: [
            'reduce' if t == ')' else 'shift' for t in parse if t != '('])
        self.build_vocab([['reduce'], ['shift']])

class ParsedTextField(data.Field):
    '''
    Modified from torchtext v0.2.3 as this class
    has been depricated in torchtext v0.4
    '''
    def __init__(self, eos_token='<pad>', lower=False):
        super(ParsedTextField, self).__init__(
            eos_token=eos_token, lower=lower, preprocessing=lambda parse: [
                t for t in parse if t not in ('(', ')')],
            postprocessing=lambda parse, _: [
                list(reversed(p)) for p in parse])

def main():
    device_type = 'cuda' if args.cuda else 'cpu'
    device = torch.device(device_type)

    print("Running on: {}".format(device))

    #####################################
    ## configure experiment parameters ##
    #####################################
    batch_sizes = [1, 32, 64, 128, 256, 512, 1024]
    epochs = 1
    learning_rate = 0.001
    max_samples = 5000 # number of samples to use for experiment
    #####################################

    inputs = ParsedTextField(lower=True)
    transitions = ShiftReduceField()
    labels = data.Field(sequential=False)

    print("Loading dataset...")
    train, dev, test = datasets.SNLI.splits(inputs, labels, transitions)
    inputs.build_vocab(train, dev, test)
    labels.build_vocab(train)
    print("Done.")
    for batch_size in batch_sizes:
        print("Batching dataset into mini-batches of size {}..".format(batch_size))
        train_iter, _, _ = data.BucketIterator.splits((train, dev, test), batch_size=batch_size, device=device)
        print("Done.")

        print("Configuring SPINN model...")
        model = SPINN(3, 500, len(inputs.vocab))
        if args.cuda:
            model.to(device)
        criterion = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=learning_rate)
        print("Done.")

        for epoch in range(epochs):
            print("starting epoch {}".format(epoch))

            all_batch_times = []
            for batch_idx, batch in enumerate(train_iter):
                opt.zero_grad() # reset gradients per mini-batch
                all_logits, all_labels = [], []

                if args.dynamic:
                    fold = torchfold.Fold()
                    if args.cuda:
                        fold.cuda()

                start = timer()
                tree_sizes = []
                # becuase batch.dataset starts at the begninning of the entire dataset
                # instead of where the previous batch left off
                for sample_idx in range(batch_idx*batch_size, (batch_idx+1)*batch_size):
                    # HACK this is to account for the final batch which may or may not be
                    # of size batch_size - there should be a more elegant solution to this
                    if sample_idx == len(batch.dataset)-1:
                        break

                    tree = Tree(batch.dataset[sample_idx].label, batch.dataset[sample_idx].premise_transitions, batch.dataset[sample_idx].premise, inputs.vocab, labels.vocab)
                    if args.dynamic:
                        all_logits.append(encode_tree_fold(fold, tree))
                    else:
                        all_logits.append(encode_tree_regular(model, tree))
                    all_labels.append(tree.label)

                if args.dynamic:
                    res = fold.apply(model, [all_logits, all_labels])
                    batch_time = timer() - start
                    loss = criterion(res[0], res[1])
                else:
                    test = np.asarray(all_labels, dtype=int)
                    x = torch.from_numpy(test).to(device)
                    batch_time = timer() - start
                    loss = criterion(torch.cat(all_logits, 0), x)

                loss.backward(); opt.step()

                ####################
                ## Gather results ##
                ####################
                all_batch_times.append(batch_time)
                results['time'].append(batch_time)
                results['epoch'].append(epoch)
                results['batch'].append(batch_idx)
                results['sample'].append(sample_idx)
                results['batch_size'].append(batch_size)
                ts = tree_size(tree.root)
                tree_sizes.append(ts)
                results['treesize'].append(np.mean(tree_sizes))
                ####################

                if batch_idx % 10 == 1:
                    print("batch size: {} sample: {}/{} loss:{:4f} - Avg. Time (per batch): {:5f}s".format(batch_size, batch_idx*batch_size, max_samples, loss, np.mean(all_batch_times)))
                # only need to look at first 5000 samples for each batch
                if batch_idx * batch_size > max_samples:
                    break

            print("done epoch {}".format(epoch))

        with open(os.path.join(ROOT, "results_fold-{}-{}-{}-backup.json".format(args.dynamic, batch_size, args.cuda)), "w+") as fd:
            json.dump(results, fd)

    with open(os.path.join(ROOT, "results_fold-{}-{}-full.json".format(args.dynamic, args.cuda)), "w+") as fd:
        json.dump(results, fd)

if __name__ == "__main__":
    main()
