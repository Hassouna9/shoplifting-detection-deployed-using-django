import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=256, num_layers=2, dropout=0.5):
        super(CNN_LSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.flatten = nn.Flatten()
        self.fc_cnn = nn.Linear(512, 256)
        self.dropout_cnn = nn.Dropout(0.5)


        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc_lstm = nn.Linear(hidden_dim, num_classes)
        self.dropout_lstm = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, C, T, H, W = x.size()

        cnn_features = []

        for t in range(T):
            frame = x[:, :, t, :, :]
            feature = self.cnn(frame)
            feature = self.flatten(feature)
            feature = self.fc_cnn(feature)
            feature = self.dropout_cnn(feature)
            cnn_features.append(feature)

        cnn_features = torch.stack(cnn_features, dim=1)

        lstm_out, (hn, cn) = self.lstm(cnn_features)

        final_feature = hn[-1, :, :]

        final_feature = self.dropout_lstm(final_feature)

        out = self.fc_lstm(final_feature)

        return out
