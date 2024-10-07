import copy
import os.path as osp
import torch
import torch.nn.functional as F
import GCL.losses as L
import GCL.augmentors as A
import torch_geometric.transforms as T

from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models.contrast_model import WithinEmbedContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import WikiCS
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.act = torch.nn.PReLU()
        self.bn = torch.nn.BatchNorm1d(2 * hidden_dim, momentum=0.01)
        self.conv1 = GCNConv(input_dim, 2 * hidden_dim, cached=False)
        self.conv2 = GCNConv(2 * hidden_dim, output_dim, cached=False)

    def forward(self, x, edge_index, edge_weight=None):
        z = self.conv1(x, edge_index, edge_weight)
        z = self.bn(z)
        z = self.act(z)
        z = self.conv2(z, edge_index, edge_weight)
        return z


class ImageGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.gc1 = GCNConv(hidden_dim, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr=None):
        # x shape: (num_nodes, in_channels, height, width)
        batch_size = x.size()[0]

        # Image processing
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).view(batch_size, -1)  # (num_nodes, hidden_dim)

        # Graph processing
        x = F.relu(self.gc1(x, edge_index, edge_attr))
        x = self.gc2(x, edge_index, edge_attr)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super().__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2


def train(encoder_model, contrast_model, data, optimizer, scheduler, val_mask):
    # Train
    train_mask = ~val_mask
    encoder_model.train()
    optimizer.zero_grad()
    _, z1_train, z2_train = encoder_model(
        data.x[train_mask], data.edge_index, data.edge_attr)
    train_loss = contrast_model(z1_train, z2_train)
    train_loss.backward()
    optimizer.step()

    # Validate
    encoder_model.eval()
    with torch.no_grad():
        _, z1_val, z2_val = encoder_model(data.x[val_mask], data.edge_index, data.edge_attr)
        val_loss = contrast_model(z1_val, z2_val).item()

    scheduler.step()
    return train_loss.item(), val_loss


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def auto_train_and_embed(data, output_dim, num_epochs=4000, hidden_dim=256,
                         lr=5e-4, patience=100, min_delta=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Split data into train and validation sets
    num_nodes = data.num_nodes
    num_val = int(0.1 * num_nodes)  # Use 10% of nodes for validation
    perm = torch.randperm(num_nodes)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[perm[:num_val]] = True

    aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])

    # Create model based on data shape
    if len(data.x.shape) == 2:
        gconv = GConv(data.x.shape[1], hidden_dim, output_dim).to(device)
    elif len(data.x.shape) == 4:
        # TODO: assumes images are (num_nodes, channels, height, width)
        gconv = ImageGCN(data.x.shape[1], hidden_dim, output_dim).to(device)
    else:
        raise ValueError("Invalid data shape, each node must be 1D or 3D.")
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = WithinEmbedContrast(loss=L.BarlowTwins()).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=lr)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=400,
        max_epochs=num_epochs
    )

    best_val_loss = float('inf')
    best_model = None
    counter = 0

    with tqdm(total=num_epochs, desc='(T)') as pbar:
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, val_loss = train(encoder_model, contrast_model, data,
                                         optimizer, scheduler, val_mask)

            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_model = copy.deepcopy(encoder_model.state_dict())
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

            pbar.set_postfix({'train_loss': train_loss.item(), 'val_loss': val_loss})
            pbar.update()

    # Load best model
    if best_model is not None:
        encoder_model.load_state_dict(best_model)

    encoder_model.eval()
    with torch.no_grad():
        z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)

    # Normalize embeddings between 0 and 1
    z_min = z.min(dim=0)[0]
    z_max = z.max(dim=0)[0]
    z_normalized = (z - z_min) / (z_max - z_min)

    return z_normalized, best_val_loss


def main():
    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets', 'WikiCS')
    dataset = WikiCS(path, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])

    gconv = GConv(input_dim=dataset.num_features, hidden_dim=256, output_dim=256).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = WithinEmbedContrast(loss=L.BarlowTwins()).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=5e-4)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=400,
        max_epochs=4000)

    with tqdm(total=4000, desc='(T)') as pbar:
        for _ in range(1, 4001):
            loss = train(encoder_model, contrast_model, data, optimizer)
            scheduler.step()
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, data)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()
