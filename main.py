import argparse
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

from Dataset import DatasetVirtualNode
from model import DipoleNet
from helper_functions import AverageMeter, CosineLearningRateScheduler


def train(model: nn.Module, train_dataloader: DataLoader, device, distance, optim, epoch, lr_scheduler):
    model.train()
    average_meter = AverageMeter()

    for i, data in enumerate(train_dataloader):
        polarities = model(
            x=data.x.to(device),
            edge_index=data.edge_index.to(device),
            edge_attr=data.edge_attr.to(device),
            batch=data.batch.to(device)
        )
        loss = distance(polarities, data.y.to(device))
        loss.backward()
        optim.step()
        optim.zero_grad()

        lr = lr_scheduler.new_lr()
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        average_meter.step(loss=loss.item())
        if i % 1000 == 0:
            print('Loss: {loss:.5f} | Std: {std:.5f} | LR: {lr:.5f} '
                  '| Done: {i}/{total} ({percentage:.3f}%) | Epoch: {epoch}'.format(
                    loss=average_meter.average(),
                    std=average_meter.std(),
                    lr=lr,
                    i=i,
                    total=len(train_dataloader),
                    percentage=(i/len(train_dataloader)) * 100, epoch=epoch))
            average_meter.clear()


def test(model: nn.Module, test_dataloader: DataLoader, device, distance, epoch):
    model.eval()
    average_meter = AverageMeter()

    for data in test_dataloader:
        polarities = model(
            x=data.x.to(device),
            edge_index=data.edge_index.to(device),
            batch=data.batch.to(device)
        )
        loss = distance(polarities, data.y.to(device))
        average_meter.step(loss=loss.item())
    print('Test Loss: {loss:.5f} | Std: {std:.5f} | Epoch: {epoch}'.format(
        loss=average_meter.average(),
        std=average_meter.std(),
        epoch=epoch
    ))


def main(root: str):
    # Training settings
    parser = argparse.ArgumentParser(description='QM9 Dipole Regression - Simple GNN Example')
    parser.add_argument('--train-batch-size', type=int, default=16, metavar='N',
                        help='Input batch size for training. Default: 16.')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='Input batch size for testing. Default: 32.')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='Number of training epochs. Default: 100.')
    parser.add_argument('--lr', type=float, default=5.0e-4, metavar='LR',
                        help='The learning rate. Default: 1.0e-1.')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='The random seed. Default: 1.')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='Whether the current model should be saved.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # Define dataset loaders
    train_dataset = DatasetVirtualNode(root=root, train=True)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0
    )
    test_dataset = DatasetVirtualNode(root=root, train=False)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=0
    )

    model = DipoleNet().to(device)
    print(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params=params, lr=args.lr)
    distance = nn.L1Loss().to(device)

    lr_scheduler = CosineLearningRateScheduler(
        i_lr=args.lr,
        n_batches_warmup=5500,
        n_total_batches=(len(train_dataloader) * args.epochs)
    )

    for epoch in range(args.epochs):
        train(model=model, train_dataloader=train_dataloader, device=device,
              distance=distance, optim=optim, epoch=epoch, lr_scheduler=lr_scheduler)
        test(model=model, test_dataloader=test_dataloader, device=device,
             distance=distance, epoch=epoch)


if __name__ == '__main__':
    main(root='/data')
