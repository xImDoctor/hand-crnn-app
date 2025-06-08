# model.py
import torch
import torch.nn as nn
import torchvision.models as models

# Алфавит для русских символов и знаков
ALPHABET = """0123456789АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя:;/?!\'\".,()-=+%[]«»№ """
BLANK = "<BLANK>"

CHARS = sorted(set(ALPHABET))
VOCAB = [BLANK] + CHARS
char2idx = {c: i for i, c in enumerate(VOCAB)}
idx2char = {i: c for c, i in char2idx.items()}

class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256, lstm_layers=2):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-3]  # C=256, H≈8, W≈128
        self.backbone = nn.Sequential(*modules)
        self.conv = nn.Conv2d(256, hidden_size, kernel_size=3, stride=1, padding=1)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=lstm_layers,
                           bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):  # x: [B, 1, H, W]
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)  # в RGB
        features = self.backbone(x)      # [B, 256, H', W']
        conv_out = self.conv(features)   # [B, hidden, H', W']
        B, C, H, W = conv_out.size()
        conv_out = conv_out.mean(2)      # [B, C, W]
        conv_out = conv_out.permute(0, 2, 1)  # [B, W, C]
        rnn_out, _ = self.rnn(conv_out)  # [B, W, 2*hidden]
        logits = self.classifier(rnn_out)  # [B, W, num_classes]
        return logits.permute(1, 0, 2)   # [W, B, C] — формат для CTC


def ctc_greedy_decode(preds, idx2char):
    """
    preds — список списков индексов [B, T], возвращает список строк
    """
    decoded_texts = []
    for seq in preds:
        tokens = []
        prev = None
        for idx in seq:
            if idx != prev and idx != 0:
                tokens.append(idx2char.get(idx, ""))
            prev = idx
        decoded_texts.append("".join(tokens))
    return decoded_texts
