import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lovasz_losses as L
from GenDatasetClass import FarmDataset,valDataset
from model import segnet_bn_relu as segnet
from unet_model import unet,nested_unet
from deeplabv3 import DeepLabV3
import time
from PIL import Image
import matplotlib.pyplot as plt
# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
#     running_loss = 0.0
#     running_correct = 0.0
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         # print(target.shape)
#         optimizer.zero_grad()
#         output = model(data)
#         # print('output size',output.size(),output)
#
#         output = F.log_softmax(output, dim=1)
#         loss = nn.NLLLoss2d(weight=torch.Tensor([0.1, 0.5, 0.5, 0.2]).to(device))(output, target)
#         running_loss += loss.item()
#         r = torch.argmax(output[0], 0).byte()
#         tg = target.byte().squeeze(0)
#
#         tmp = 0
#         count = 0
#         for i in range(1, 4):
#             mp = r == i
#             tr = tg == i
#             tp = mp * tr == 1
#             t = (mp + tr - tp).sum().item()
#             if t == 0:
#                 continue
#             else:
#                 tmp += tp.sum().item() / t
#                 count += 1
#         if count > 0:
#             running_correct += tmp / count
#         loss.backward()
#
#         optimizer.step()
#
#         # time.sleep(0.6)#make gpu sleep
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                        100. * batch_idx / len(train_loader), loss.item()))
#     if epoch % 2 == 0:
#         imgd = output.detach()[0, :, :, :].cpu()
#         img = torch.argmax(imgd, 0).byte().numpy()
#         imgx = Image.fromarray(img).convert('L')
#         imgxx = Image.fromarray(target.detach()[0, :, :].cpu().byte().numpy() * 255).convert('L')
#         imgx.save("./tmp/predict{}.bmp".format(epoch))
#         imgxx.save('./tmp/real{}.bmp'.format(epoch))
#
#     loss = running_loss/len(train_loader.dataset)
#     correct = running_correct/len()

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    running_correct = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        print('data',data.shape)
        print('target',target.shape)
        optimizer.zero_grad()
        output = model(data)
        output = F.log_softmax(output, dim=1)
        # print('output size',output.shape)
        loss = nn.NLLLoss2d(weight=torch.Tensor([0.1, 0.5, 0.5, 0.2]).to(device))(output, target)
        # out = F.softmax(output, dim=1)
        # loss = L.lovasz_softmax(output, target)
        running_loss += loss.item()

        maxbatch = len(data)
        for idx_train in range(maxbatch):
            out = output[idx_train]
            print('out',out.shape)
            r = torch.argmax(out, 0).byte()
            print('r',r.shape)
            tg = target[idx_train].byte().squeeze(0)
            print('tg',tg.shape)
            tmp = 0
            count = 0
            for i in range(1, 4):
                mp = r == i
                tr = tg == i
                tp = mp * tr == 1
                t = (mp + tr - tp).sum().item()
                if t == 0:
                    continue
                else:
                    tmp += tp.sum().item() / t
                    count += 1
            if count > 0:
                running_correct += tmp / count

        loss.backward()

        optimizer.step()

        # time.sleep(0.6)#make gpu sleep
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tmiou: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),running_correct/len(data)))
    if epoch % 2 == 0:
        imgd = output.detach()[0, :, :, :].cpu()
        img = torch.argmax(imgd, 0).byte().numpy()
        imgx = Image.fromarray(img).convert('L')
        imgxx = Image.fromarray(target.detach()[0, :, :].cpu().byte().numpy() * 255).convert('L')
        imgx.save("./tmp/predict{}.bmp".format(epoch))
        imgxx.save('./tmp/real{}.bmp'.format(epoch))

    loss = running_loss / len(train_loader.dataset)
    correct = running_correct / len(train_loader.dataset)
    return loss,correct


def val(args, model, device, testdataset, issave=False):
    model.eval()
    test_loss = 0
    correct = 0
    evalid = [i for i in range(2)]
    maxbatch = len(evalid)
    with torch.no_grad():
        for idx in evalid:
            data, target = testdataset[idx]
            data, target = data.unsqueeze(0).to(device), target.unsqueeze(0).to(device)
            # print(target.shape)
            target = target[:, :1472, :1472]
            output = model(data[:, :, :1472, :1472])
            output = F.log_softmax(output, dim=1)
            loss = nn.NLLLoss2d().to('cuda')(output, target)
            # out = F.softmax(output, dim=1)
            # loss = L.lovasz_softmax(output, target)
            test_loss += loss.item()
            # print('val output',output[0].shape)
            r = torch.argmax(output[0], 0).byte()
            # print('val r',r.shape)
            tg = target.byte().squeeze(0)

            tmp = 0
            count = 0
            for i in range(1, 4):
                mp = r == i
                tr = tg == i
                tp = mp * tr == 1
                t = (mp + tr - tp).sum().item()
                if t == 0:
                    continue
                else:
                    tmp += tp.sum().item() / t
                    count += 1
            if count > 0:
                correct += tmp / count

            if issave:
                Image.fromarray(r.cpu().numpy()).save('predict.png')
                Image.fromarray(tg.cpu().numpy()).save('target.png')
                input()
    test_loss = test_loss / maxbatch
    correct = correct/maxbatch
    print('Test Loss is {:.6f}, \tmean IOU is: {:.4f}%'.format(test_loss, correct))
    return test_loss,correct

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Scratch segmentation Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('my device is :', device)
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(FarmDataset(istrain=True), batch_size=args.batch_size, shuffle=True,
                                               drop_last=True,**kwargs)

    startepoch = 0
    # model = torch.load('./tmp/model{}'.format(startepoch)) if startepoch else segnet(3, 4).to(device)
    # model = torch.load('./tmp/model{}'.format(startepoch)) if startepoch else unet(3, 4).to(device)
    # model = torch.load('./tmp/model{}'.format(startepoch)) if startepoch else nested_unet(3, 4).to(device)
    model = torch.load('./tmp/model{}'.format(startepoch)) if startepoch else DeepLabV3().to(device)

    args.epochs = 5
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses , train_mious = [],[]
    val_losses , val_mious = [],[]
    for epoch in range(startepoch, args.epochs + 1):
        #relu+bn

        train_loss,train_miou = train(args, model, device, train_loader, optimizer, epoch)
        val_loss,val_miou = val(args, model, device, valDataset(istrain=True, isaug=False), issave=False)
        train_losses.append(train_loss)
        train_mious.append(train_miou)
        val_losses.append(val_loss)
        val_mious.append(val_miou)
        if epoch % 3 == 0:
            print(epoch)
            val(args, model, device, valDataset(istrain=True, isaug=False), issave=False)
            torch.save(model, './tmp/model{}'.format(epoch))

    plt.plot(range(1,len(train_losses)+1),train_losses,'bo',label = 'train loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r', label='val loss')
    plt.legend()
    plt.show()
    plt.plot(range(1, len(train_mious) + 1), train_mious, 'bo', label='train miou')
    plt.plot(range(1, len(val_mious) + 1), val_mious, 'r', label='val miou')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()