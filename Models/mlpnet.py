import torch.nn as nn
import torch.nn.functional as F

from Layers import layers

class MlpNet(nn.Module):
    def __init__(self, input_size, num_hidden_nodes1=40, num_hidden_nodes2=20, num_classes=10, enable_dropout=False, disable_bias=True, disable_log_soft=False):
        super(MlpNet, self).__init__()
        assert isinstance(input_size, int), "input_size must be an integer"
        assert isinstance(num_hidden_nodes1, int), "num_hidden_nodes1 must be an integer"
        assert isinstance(num_hidden_nodes2, int), "num_hidden_nodes2 must be an integer"
        assert isinstance(num_classes, int), "num_classes must be an integer"

        self.enable_dropout = enable_dropout
        self.do_log_soft = not disable_log_soft
        self.fc1 = layers.Linear(input_size, num_hidden_nodes1, bias=not disable_bias)
        self.fc2 = layers.Linear(num_hidden_nodes1, num_hidden_nodes2, bias=not disable_bias)
        self.fc3 = layers.Linear(num_hidden_nodes2, num_classes, bias=not disable_bias)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        
        if self.do_log_soft:
            return F.log_softmax(x, dim=1)
        else:
            return x

def mlpnet_basic(input_shape, num_classes, dense_classifier=False, pretrained=False):
    #cifar-10 = 3072
    return MlpNet(3072, num_classes=num_classes)

