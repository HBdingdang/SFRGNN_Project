
class DomainDiscriminator(nn.Module):
    def __init__(self, in_feature: int, hidden_size: int, batch_norm=True, use_sigmoid=True, dropout_rate=0.5):
        super(DomainDiscriminator, self).__init__()

        self.fc1 = nn.Linear(in_feature, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

        # 批归一化
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)
            self.bn3 = nn.BatchNorm1d(hidden_size)

        self.use_sigmoid = use_sigmoid
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout_rate)

        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        if self.batch_norm:
            x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc4(x)

        if self.use_sigmoid:
            x = self.sigmoid(x)

        return x
