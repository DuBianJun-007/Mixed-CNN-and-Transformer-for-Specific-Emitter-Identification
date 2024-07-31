import yaml
import numpy as np
import time
import argparse
from pathlib import Path
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from modules.model import model
from datasets.Bluetooth.Bluetooth_Dataset import BluetoothDataset
from sklearn.metrics import confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    '''train'''
    parser.add_argument("--max_lr", default=3e-4, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--run_name", default='20Ghz', type=Path)
    parser.add_argument('--loss_type', default="cross_entropy", type=str)
    parser.add_argument('--n_epochs', default=3500, type=int)
    parser.add_argument('--save_path', default=Path(r'outputs'), type=Path)
    parser.add_argument('--scheduler', default=None, type=str)
    '''data'''
    # if args.dataset == 'Bluetooth_250MHz':
    #     args.data_path = r'./datasets/Bluetooth/labels_half_step.xlsx'
    #     args.n_classes = 33
    #     args.sheet_name = 'Dataset 250 Msps -IQ'
    # elif args.dataset == 'Bluetooth_20GHz':
    #     args.data_path = r'./datasets/Bluetooth/labels_half_step.xlsx'
    #     args.n_classes = 22
    #     args.sheet_name = 'Dataset 20 Gsps -IQ'
    # elif args.dataset == 'Bluetooth_10GHz':
    #     args.data_path = r'./datasets/Bluetooth/labels_half_step.xlsx'
    #     args.n_classes = 16
    #     args.sheet_name = 'Dataset 10 Gsps -IQ'
    # elif args.dataset == 'Bluetooth_5GHz':
    #     args.data_path = r'./datasets/Bluetooth/labels_half_step.xlsx'
    #     args.n_classes = 17
    #     args.sheet_name = 'Dataset 5 Gsps - IQ'
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seq_len', default=50000, type=int)
    parser.add_argument('--dataset', default="Bluetooth_20GHz", type=str)
    parser.add_argument("--n_classes", default=22, type=int)
    parser.add_argument("--data_path", default=Path(r'./datasets/Bluetooth/labels_half_step.xlsx'), type=Path)
    parser.add_argument("--sheet_name", default='Dataset 20 Gsps -IQ', type=str)
    parser.add_argument("--load_in_memory", default=False, type=bool)
    '''net'''
    parser.add_argument('--ds_factors', nargs='+', type=int, default=[4, 4, 4, 4])
    parser.add_argument('--n_head', default=8, type=int)
    parser.add_argument('--n_layers', default=2, type=int)
    parser.add_argument("--emb_dim", default=128, type=int)
    parser.add_argument("--nf", default=32, type=int)
    parser.add_argument("--dim_feedforward", default=128, type=int)

    args = parser.parse_args()
    return args


def dummy_run(net, batch_sz, seq_len):
    print("***********Dummy Run************")
    d = next(net.parameters()).device
    x = torch.randn(batch_sz, 1, 2, seq_len, device=d, requires_grad=False)
    t_batch = time.time()
    with torch.no_grad():
        for k in range(1):
            _ = net(x)
    t_batch = (time.time() - t_batch) / 10
    print("dummy succededd, avg_time_batch:{}ms".format(t_batch * 1000))
    del x
    return True


def create_dataset(args):
    train_set, test_set = None, None
    if args.dataset == 'Bluetooth_20GHz' or args.dataset == 'Bluetooth_10GHz' or args.dataset == 'Bluetooth_5GHz' or \
            args.dataset == 'Bluetooth_250MHz':
        train_set = BluetoothDataset(
            args.data_path,
            args.sheet_name,
            mode='train',
            load_in_memory=args.load_in_memory
        )

        test_set = BluetoothDataset(
            args.data_path,
            args.sheet_name,
            mode='val',
            load_in_memory=args.load_in_memory
        )

    return train_set, test_set


def create_model(args):
    ds_fac = np.prod(np.array(args.ds_factors)) * 4  # args.ds_factors = [4, 4, 4, 4]
    net = model(nf=args.nf,
                dim_feedforward=args.dim_feedforward,
                clip_length=args.seq_len // ds_fac,
                embed_dim=args.emb_dim,
                n_layers=args.n_layers,
                nhead=args.n_head,
                n_classes=args.n_classes,
                factors=args.ds_factors,
                )
    return net


def save_model(net, opt, loss, acc, steps, root, scaler=None):
    chkpnt = {
        'model_dict': net.state_dict(),
        'opt_dict': opt.state_dict(),
        'steps': steps,
    }
    if scaler is not None:
        chkpnt['scaler'] = scaler.state_dict()
    torch.save(chkpnt, root / "chkpnt.pt")
    torch.save(net.state_dict(), root / "best_model.pt")
    print(acc, loss, 'saved')
    return True


def train(args):
    #######################
    # Create data loaders #
    #######################
    train_set, test_set = create_dataset(args)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              shuffle=True,
                              )
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             shuffle=False,
                             )

    #####################
    # Network           #
    #####################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = create_model(args)
    net.to(device)

    # Load the pre-trained weights
    # checkpoint = torch.load('./best_models/250MHz/best_model.pth', map_location='cuda:0')
    # net.load_state_dict(checkpoint)

    #####################
    # optimizer         #
    #####################
    optimizer = torch.optim.Adam(net.parameters(), lr=args.max_lr, weight_decay=1e-4)
    #####################
    # losses            #
    #####################
    if args.loss_type == "label_smooth":
        from modules.losses import LabelSmoothCrossEntropyLoss
        criterion = LabelSmoothCrossEntropyLoss(smoothing=0.1, reduction='sum').to(device)
    elif args.loss_type == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)
    elif args.loss_type == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum').to(device)
    else:
        raise ValueError

    ####################################
    # Dump arguments and create logger #
    ####################################
    root = args.save_path / args.run_name
    root.mkdir(parents=True, exist_ok=True)

    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)
    print(args)
    writer = SummaryWriter(str(root))
    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    dummy_run(net, args.batch_size, args.seq_len)
    net.train()
    best_acc = 0.0
    num_epochs = args.n_epochs
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_top1_acc, train_top5_acc, train_accuracy, train_precision, train_recall, train_f1 = train_one_epoch(
            net, train_loader, criterion, optimizer, device)
        test_loss, test_acc, test_top1_acc, test_top5_acc, test_accuracy, test_precision, test_recall, test_f1 = evaluate(
            net, test_loader, criterion, device)

        print('Frequence:20GHz')
        print(
            f'Epoch {epoch}/{num_epochs} Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} Top-1 Acc: {train_top1_acc:.4f} Top-5 Acc: {train_top5_acc:.4f}')
        print(
            f'Epoch {epoch}/{num_epochs} Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} Top-1 Acc: {test_top1_acc:.4f} Top-5 Acc: {test_top5_acc:.4f}')
        print(
            f'Train Accuracy: {train_accuracy:.4f} Precision: {train_precision:.4f} Recall: {train_recall:.4f} F1 Score: {train_f1:.4f}')
        print(
            f'Test Accuracy: {test_accuracy:.4f} Precision: {test_precision:.4f} Recall: {test_recall:.4f} F1 Score: {test_f1:.4f}')

        scheduler.step()

        # 记录训练和验证的损失及Top-1精度
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', test_loss, epoch)
        writer.add_scalar('Accuracy/Top1/train', train_top1_acc, epoch)
        writer.add_scalar('Accuracy/Top1/val', test_top1_acc, epoch)
        writer.add_scalar('Metrics/Train/Accuracy', train_accuracy, epoch)
        writer.add_scalar('Metrics/Train/Precision', train_precision, epoch)
        writer.add_scalar('Metrics/Train/Recall', train_recall, epoch)
        writer.add_scalar('Metrics/Train/F1', train_f1, epoch)
        writer.add_scalar('Metrics/Test/Accuracy', test_accuracy, epoch)
        writer.add_scalar('Metrics/Test/Precision', test_precision, epoch)
        writer.add_scalar('Metrics/Test/Recall', test_recall, epoch)
        writer.add_scalar('Metrics/Test/F1', test_f1, epoch)

        # 如果当前的模型在验证集上的准确率比之前的模型好，那么保存当前的模型
        if test_acc > best_acc:
            best_acc = test_acc
            print(f'best_acc: {best_acc}')
            torch.save(net.state_dict(), './best_models/20GHz/best_model.pth')  # 保存最好的模型权重到文件

    writer.close()


# 计算指标函数
def calculate_metrics(labels, preds):
    cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    return accuracy, precision, recall, f1_score


# 定义训练函数
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    top1_corrects = 0
    top5_corrects = 0

    all_labels = []
    all_preds = []

    for inputs, labels in train_loader:
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        # 计算Top-1和Top-5精度
        _, top5_preds = outputs.topk(5, 1, True, True)
        top1_corrects += torch.sum(preds == labels.data)
        top5_corrects += torch.sum(top5_preds == labels.unsqueeze(1).data)

        all_labels.append(labels)
        all_preds.append(preds)

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    top1_acc = top1_corrects.double() / len(train_loader.dataset)
    top5_acc = top5_corrects.double() / len(train_loader.dataset)

    accuracy, precision, recall, f1_score = calculate_metrics(all_labels, all_preds)

    return epoch_loss, epoch_acc, top1_acc, top5_acc, accuracy, precision, recall, f1_score


# 在评估函数中
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    top1_corrects = 0
    top5_corrects = 0

    all_labels = []
    all_preds = []

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        # 计算Top-1和Top-5精度
        _, top5_preds = outputs.topk(5, 1, True, True)
        top1_corrects += torch.sum(preds == labels.data)
        top5_corrects += torch.sum(top5_preds == labels.unsqueeze(1).data)

        all_labels.append(labels)
        all_preds.append(preds)

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects.double() / len(test_loader.dataset)
    top1_acc = top1_corrects.double() / len(test_loader.dataset)
    top5_acc = top5_corrects.double() / len(test_loader.dataset)

    accuracy, precision, recall, f1_score = calculate_metrics(all_labels, all_preds)

    return epoch_loss, epoch_acc, top1_acc, top5_acc, accuracy, precision, recall, f1_score


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
