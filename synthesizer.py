import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import re


# https://discuss.pytorch.org/t/calculating-flops-of-a-given-pytorch-model/3711
def get_num_gen(gen):
    return sum(1 for x in gen)


def flops_layer(layer):
    idx_type_end = layer.find('(')
    type_name = layer[:idx_type_end]

    params = re.findall('[^a-z](\d+)', layer)
    flops = 1

    if layer.find('Linear') >= 0:
        C1 = int(params[0])
        C2 = int(params[1])
        flops = C1 * C2

    elif layer.find('Conv2d') >= 0:
        C1 = int(params[0])
        C2 = int(params[1])
        K1 = int(params[2])
        K2 = int(params[3])

        # image size
        H = 32
        W = 32
        flops = C1 * C2 * K1 * K2 * H * W

    #     print(type_name, flops)
    return flops


def calculate_flops(gen):
    flops = 0

    for child in gen:
        num_children = get_num_gen(child.children())
        # leaf node
        if num_children == 0:
            flops += flops_layer(str(child))

        else:
            flops += calculate_flops(child.children())

    return flops


# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/5
def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class Transformer(nn.Module):
    def __init__(self, in_dims):
        super(Transformer, self).__init__()
        self.temperature = in_dims ** 0.5
        self.query_fc = nn.Linear(in_dims, in_dims)
        self.key_fc = nn.Linear(in_dims, in_dims)
        self.value_fc = nn.Linear(in_dims, in_dims)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query_fc(x)
        key = self.key_fc(x).permute(0, 2, 1)

        energy = torch.bmm(query / self.temperature, key)
        attention = self.softmax(energy)

        value = self.value_fc(x)

        out = torch.bmm(attention, value)

        return out, attention


class SynthesizerDense(nn.Module):
    def __init__(self, in_dims, sentence_length):
        super(SynthesizerDense, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(in_dims, sentence_length),
            nn.ReLU(),
            nn.Linear(sentence_length, sentence_length)
        )

        self.value_fc = nn.Linear(in_dims, in_dims)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        energy = self.dense(x)
        attention = self.softmax(energy)

        value = self.value_fc(x)

        out = torch.bmm(attention, value)

        return out, attention


class SynthesizerRandom(nn.Module):
    def __init__(self, in_dims, sentence_length, fixed=False):
        super(SynthesizerRandom, self).__init__()
        if fixed:
            self.attention = nn.Parameter(torch.empty(1, sentence_length, sentence_length), requires_grad=False)
        else:
            self.attention = nn.Parameter(torch.empty(1, sentence_length, sentence_length), requires_grad=True)
        nn.init.xavier_uniform_(self.attention)

        self.value_fc = nn.Linear(in_dims, in_dims)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        value = self.value_fc(x)
        out = torch.matmul(self.attention, value)

        return out, self.attention


class FactorizedSynthesizerDense(nn.Module):
    def __init__(self, in_dims, sentence_length):
        super(FactorizedSynthesizerDense, self).__init__()
        self.a = 4
        self.b = sentence_length // self.a

        self.a_proj = nn.Linear(in_dims, self.a)
        self.b_proj = nn.Linear(in_dims, self.b)

        self.value_fc = nn.Linear(in_dims, in_dims)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        A = self.a_proj(x).repeat([1, 1, self.b])
        B = self.b_proj(x).repeat([1, 1, self.a])

        energy = A * B
        attention = self.softmax(energy)

        value = self.value_fc(x)

        out = torch.bmm(attention, value)
        return out, attention


class FactorizedSynthesizerRandom(nn.Module):
    def __init__(self, in_dims):
        super(FactorizedSynthesizerRandom, self).__init__()
        self.k = 8
        self.query_fc = nn.Linear(in_dims, self.k)
        self.key_fc = nn.Linear(in_dims, self.k)
        self.value_fc = nn.Linear(in_dims, in_dims)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query_fc(x)
        key = self.key_fc(x).permute(0, 2, 1)

        energy = torch.bmm(query, key)
        attention = self.softmax(energy)

        value = self.value_fc(x)

        out = torch.bmm(attention, value)

        return out, attention


class MixtureSynthesizers(nn.Module):
    def __init__(self, in_dims, sentence_length):
        super(MixtureSynthesizers, self).__init__()
        # Random + Vanilla

        # Random
        self.attention = nn.Parameter(torch.empty(1, sentence_length, sentence_length), requires_grad=True)
        nn.init.xavier_uniform_(self.attention)

        # Vanilla
        self.query_fc = nn.Linear(in_dims, in_dims)
        self.key_fc = nn.Linear(in_dims, in_dims)

        self.value_fc = nn.Linear(in_dims, in_dims)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query_fc(x)
        key = self.key_fc(x).permute(0, 2, 1)

        vanilla_energy = torch.bmm(query, key)
        energy = self.attention + vanilla_energy
        attention = self.softmax(energy)

        value = self.value_fc(x)

        out = torch.bmm(attention, value)

        return out, attention


def main():
    batch_size, channel_dim, sentence_length = 2, 1024, 32
    x = torch.randn([batch_size, sentence_length, channel_dim])

    vanilla = Transformer(channel_dim)
    out, attention_map = vanilla(x)
    print(out.size(), attention_map.size())
    n_params, flops = get_n_params(vanilla), calculate_flops(vanilla.children())
    print('vanilla, n_params: {}, flops: {}'.format(n_params, flops))

    dense_synthesizer = SynthesizerDense(channel_dim, sentence_length)
    out, attention_map = dense_synthesizer(x)
    print(out.size(), attention_map.size())
    n_params, flops = get_n_params(dense_synthesizer), calculate_flops(dense_synthesizer.children())
    print('dense_synthesizer, n_params: {}, flops: {}'.format(n_params, flops))

    random_synthesizer = SynthesizerRandom(channel_dim, sentence_length)
    out, attention_map = random_synthesizer(x)
    print(out.size(), attention_map.size())
    n_params, flops = get_n_params(random_synthesizer), calculate_flops(random_synthesizer.children())
    print('random_synthesizer, n_params: {}, flops: {}'.format(n_params, flops))

    random_synthesizer_fix = SynthesizerRandom(channel_dim, sentence_length, fixed=True)
    out, attention_map = random_synthesizer_fix(x)
    print(out.size(), attention_map.size())
    n_params, flops = get_n_params(random_synthesizer_fix), calculate_flops(random_synthesizer_fix.children())
    print('random_synthesizer_fix, n_params: {}, flops: {}'.format(n_params, flops))

    factorized_synthesizer_random = FactorizedSynthesizerRandom(channel_dim)
    out, attention_map = factorized_synthesizer_random(x)
    print(out.size(), attention_map.size())
    n_params, flops = get_n_params(factorized_synthesizer_random), calculate_flops(factorized_synthesizer_random.children())
    print('factorized_synthesizer_random, n_params: {}, flops: {}'.format(n_params, flops))

    factorized_synthesizer_dense = FactorizedSynthesizerDense(channel_dim, sentence_length)
    out, attention_map = factorized_synthesizer_dense(x)
    print(out.size(), attention_map.size())
    n_params, flops = get_n_params(factorized_synthesizer_dense), calculate_flops(factorized_synthesizer_dense.children())
    print('factorized_synthesizer_dense, n_params: {}, flops: {}'.format(n_params, flops))

    mixture_synthesizer = MixtureSynthesizers(channel_dim, sentence_length)
    out, attention_map = mixture_synthesizer(x)
    print(out.size(), attention_map.size())
    n_params, flops = get_n_params(mixture_synthesizer), calculate_flops(mixture_synthesizer.children())
    print('mixture_synthesizer, n_params: {}, flops: {}'.format(n_params, flops))


if __name__ == '__main__':
    main()
