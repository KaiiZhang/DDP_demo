import os
from datetime import datetime
import argparse
# import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--epochs', default=1, type=int, 
                        metavar='N',
                        help='number of total epochs to run')

    args = parser.parse_args()
    train(args.local_rank, args)


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def evaluate(model, gpu, test_loader, rank):
    model.eval()
    size = torch.tensor(0.).to(gpu)
    correct = torch.tensor(0.).to(gpu)
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.to(gpu)
            labels = labels.to(gpu)
            # Forward pass
            outputs = model(images)
            size += images.shape[0]
            correct += (outputs.argmax(1) == labels).type(torch.float).sum()
    dist.reduce(size, 0, op=dist.ReduceOp.SUM)
    dist.reduce(correct, 0, op=dist.ReduceOp.SUM)
    if rank==0:
        print('Evaluate accuracy is {:.2f}'.format(correct / size))

def train(gpu, args):
    print('start init process group')
    print('local rank: {}'.format(args.local_rank))
    dist.init_process_group(backend='nccl', init_method='env://')
    world_size =  dist.get_world_size()
    global_rank = dist.get_rank()
    print('rank {}/{} is ready'.format(global_rank, world_size))
    dist.barrier(device_ids=[gpu])
    torch.manual_seed(0) # set random seed in eyery process
    model = ConvNet()
    model.cuda(gpu)
    batch_size = 10 
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Wrap the model
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 
    print('construct model in {}/{}'.format(global_rank, world_size))
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=args.local_rank, broadcast_buffers=False)
    # Data loading code
    print('construct dataloader in {}/{}'.format(global_rank, world_size))
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=False,
                                               transform=transforms.ToTensor(),
                                               download=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=test_sampler)

    start = datetime.now()
    total_step = len(train_loader) # change to orignal_length / args.world_size
    print('start train in {}/{}'.format(global_rank, world_size))
    for epoch in range(args.epochs):
        model.train()
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(gpu)
            labels = labels.to(gpu)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and global_rank == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                   loss.item()))
        evaluate(model, gpu, test_loader, global_rank)        
    dist.destroy_process_group()                                                      
    if global_rank == 0:
        print("Training complete in: " + str(datetime.now() - start))


if __name__ == '__main__':
    main()