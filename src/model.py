import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

class TcgaNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(TcgaNet, self).__init__()


        #input to hidden
        self.layer1 = nn.Linear(input_dim, hidden_dim)

        # activation
        self.relu = nn.ReLU()

        # dropout
        self.dropout = nn.Dropout(p=dropout)

        #hidden to output
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

if __name__ == "__main__":

    model = TcgaNet(input_dim=30865, hidden_dim=128, output_dim=5, dropout=0.3)

    #fake test
    data = torch.rand(16, 30865)
    output = model(data)

    print(f"Output shape: {output.shape}")
