import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from utils.rnn import RNNLayer
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

"GAT Module"

class Encoder(nn.Module):
    def __init__(self, din=32, hidden_dim=64):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(din, hidden_dim)

    def forward(self, x):
        embedding = F.relu(self.fc(x))
        return embedding


class AttModel(nn.Module):
    def __init__(self, n_node, din, hidden_dim, dout):
        super(AttModel, self).__init__()
        self.fcv = nn.Linear(din, hidden_dim)
        self.fck = nn.Linear(din, hidden_dim)
        self.fcq = nn.Linear(din, hidden_dim)
        self.fcout = nn.Linear(hidden_dim, dout)

    def forward(self, x, mask):
        v = F.relu(self.fcv(x))
        q = F.relu(self.fcq(x))
        k = F.relu(self.fck(x)).permute(0, 2, 1)
        att = F.softmax(torch.mul(torch.bmm(q, k), mask) - 9e15 * (1 - mask), dim=2)
        out = torch.bmm(att, v)
        out = F.relu(self.fcout(out))
        return out


class Q_Net(nn.Module):
    def __init__(self, hidden_dim, dout):
        super(Q_Net, self).__init__()
        self.fc = nn.Linear(hidden_dim, dout)

    def forward(self, x):
        q = self.fc(x)
        return q


class DGN(nn.Module):
    def __init__(self, n_agent, num_inputs, hidden_dim, num_actions, args):
        super(DGN, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self._recurrent_N = args.recurrent_N
        self.encoder = Encoder(num_inputs, hidden_dim)
        self.att_1 = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        self.att_2 = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        self.rnn = RNNLayer(hidden_dim, hidden_dim, self._recurrent_N, self._use_orthogonal)
        self.q_net = Q_Net(hidden_dim, 1)

    def forward(self, x, mask, rnn_states):
        h1 = self.encoder(x)
        critic_features, rnn_states = self.rnn(h1.reshape(h1.shape[0] * h1.shape[1], h1.shape[2]), rnn_states)
        critic_features = critic_features.reshape(-1, x.shape[1], critic_features.shape[1])
        h2 = self.att_1(critic_features, mask)
        h3 = self.att_2(h2, mask).reshape(-1, 64)
        values = self.q_net(h3)

        return values, rnn_states
