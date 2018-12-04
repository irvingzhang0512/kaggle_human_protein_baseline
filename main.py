import os
import time
import torch
import random
import warnings
import numpy as np
import pandas as pd
import sys
import argparse

from config import config
from utils import save_checkpoint, AverageMeter, Logger, FocalLoss, get_learning_rate, time_to_str, F1Meter, F1Loss
from data import HumanDataset
from tqdm import tqdm
from datetime import datetime
from models.model import get_net
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer

# set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

# set gpu settings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.backends.cudnn.benchmark = True

# other settings
warnings.filterwarnings('ignore')
logging_pattern = '%s %5.1f %6.1f    |      %.3f  %.3f       |      %.3f  %.4f      |       %s  %s        | %s'

# # 0.6717
# thresholds = 0.15

# # 0.6870
# thresholds = 0.2

# 0.7096
thresholds = 0.4

# # 0.6646
# thresholds = np.array([0.407, 0.441, 0.161, 0.145, 0.299, 0.129, 0.25, 0.414, 0.01, 0.028, 0.021, 0.125,
#                        0.113, 0.387, 0.602, 0.001, 0.137, 0.199, 0.176, 0.25, 0.095, 0.29, 0.159, 0.255,
#                        0.231, 0.363, 0.117, 0., ])


# # 0.7038
# thresholds = np.array([0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.4,
#                        0.4, 0.4, 0.4, 0.2, 0.4, 0.4, 0.4, 0.4, 0.2, 0.4, 0.4, 0.4,
#                        0.4, 0.4, 0.4, 0.2, ])

def train(train_loader, model, criterion, optimizer, epoch, valid_loss, best_results, start):
    losses = AverageMeter()
    f1 = F1Meter()
    model.train()
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
        output = model(images)
        loss = criterion(output, target)

        losses.update(loss.item(), images.size(0))
        f1.update(output.sigmoid().cpu() > thresholds, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % config.logging_every_n_steps == 0:
            # print('\r', end='', flush=True)
            message = logging_pattern % (
                "train", i / len(train_loader) + epoch, epoch,
                losses.avg, f1.f1,
                valid_loss[0], valid_loss[1],
                str(best_results[0])[:8], str(best_results[1])[:8],
                time_to_str((timer() - start), 'min'))
            print(message, end='\n', flush=True)
    return [losses.avg, f1.f1]


def evaluate(val_loader, model, criterion, epoch, train_loss, best_results, start):
    losses = AverageMeter()
    f1 = F1Meter()
    model.cuda()
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images_var = images.cuda(non_blocking=True)
            target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
            output = model(images_var)
            loss = criterion(output, target)
            losses.update(loss.item(), images_var.size(0))
            f1.update(output.sigmoid().cpu() > thresholds, target)
            if i % config.logging_every_n_steps == 0:
                # print('\r', end='', flush=True)
                message = logging_pattern % (
                    "val", i / len(val_loader) + epoch, epoch,
                    train_loss[0], train_loss[1],
                    losses.avg, f1.f1,
                    str(best_results[0])[:8], str(best_results[1])[:8],
                    time_to_str((timer() - start), 'min'))
                print(message, end='\n', flush=True)

    return [losses.avg, f1.f1]


def test(test_loader, model, folds):
    sample_submission_df = pd.read_csv(config.test_csv)
    labels, submissions = [], []
    model.cuda()
    model.eval()
    for i, (x, filepath) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            image_var = x.cuda(non_blocking=True)
            y_pred = model(image_var)
            cur_label = y_pred.sigmoid().cpu().data.numpy()
            labels.append(cur_label > thresholds)

    for row in np.concatenate(labels):
        subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
        submissions.append(subrow)
    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv(os.path.join(config.submit, '%s_bestloss_submission.csv' % config.model_name),
                                index=None)


def training(model, fold, args):
    # logging issues
    log = Logger()
    log.open(os.path.join(config.logs_dir, "%s_log_train.txt" % config.model_name), mode="a")
    log.write(
        "\n---------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 20))

    log.write(
        '----------------------|--------- Train ---------|-------- Valid ---------|-------Best '
        'Results-------|----------|\n')
    log.write(
        'mode   iter   epoch   |      loss   f1_macro    |      loss   f1_macro   |       loss   f1_macro    | time   '
        '  |\n')
    log.write(
        '----------------------------------------------------------------------------------------------------------'
        '----\n')

    # training params
    optimizer = optim.SGD(model.parameters(),
                          lr=config.learning_rate_start,
                          momentum=0.9,
                          weight_decay=config.weight_decay)
    # criterion = nn.BCEWithLogitsLoss().cuda()
    # criterion = FocalLoss().cuda()
    criterion = F1Loss().cuda()
    best_results = [np.inf, 0]
    val_metrics = [np.inf, 0]
    scheduler = lr_scheduler.StepLR(optimizer,
                                    step_size=config.learning_rate_decay_epochs,
                                    gamma=config.learning_rate_decay_rate)
    start = timer()

    # load dataset
    all_files = pd.read_csv(config.train_csv)
    train_data_list, val_data_list = train_test_split(all_files, test_size=config.val_percent, random_state=2050)
    train_gen = HumanDataset(train_data_list, config.train_dir, mode="train")
    train_loader = DataLoader(train_gen, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_gen = HumanDataset(val_data_list, config.train_dir, augument=False, mode="train")
    val_loader = DataLoader(val_gen, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # train
    for epoch in range(0, config.epochs):
        # training & evaluating
        scheduler.step(epoch)
        get_learning_rate(optimizer)
        train_metrics = train(train_loader, model, criterion, optimizer, epoch, val_metrics, best_results, start)
        val_metrics = evaluate(val_loader, model, criterion, epoch, train_metrics, best_results, start)

        # check results
        is_best_loss = val_metrics[0] < best_results[0]
        best_results[0] = min(val_metrics[0], best_results[0])
        is_best_f1 = val_metrics[1] > best_results[1]
        best_results[1] = max(val_metrics[1], best_results[1])

        # save model
        save_checkpoint({
            "epoch": epoch + 1,
            "model_name": config.model_name,
            "state_dict": model.state_dict(),
            "best_loss": best_results[0],
            "optimizer": optimizer.state_dict(),
            "fold": fold,
            "best_f1": best_results[1],
        }, is_best_loss, is_best_f1, fold)

        # print logs
        print('\r', end='', flush=True)
        log.write(
            logging_pattern % (
                "best", epoch, epoch,
                train_metrics[0], train_metrics[1],
                val_metrics[0], val_metrics[1],
                str(best_results[0])[:8], str(best_results[1])[:8],
                time_to_str((timer() - start), 'min')
            )
        )
        log.write("\n")
        time.sleep(0.01)


def testing(model, fold, args):
    print('start testing')
    # load dataset
    test_files = pd.read_csv(config.test_csv)
    test_gen = HumanDataset(test_files, config.test_dir, augument=False, mode="test")
    test_loader = DataLoader(test_gen, 1, shuffle=False, pin_memory=True, num_workers=4)

    # load model
    best_model = torch.load(
        "%s/%s_fold_%s_model_best_loss.pth.tar" % (config.best_models, config.model_name, str(fold)))
    # best_model = torch.load("checkpoints/bninception_bcelog/0/checkpoint.pth.tar")
    model.load_state_dict(best_model["state_dict"])
    test(test_loader, model, fold)


def evaluating(model, fold, args):
    # load model
    best_model = torch.load(
        "%s/%s_fold_%s_model_best_loss.pth.tar" % (config.best_models, config.model_name, str(fold)))
    model.load_state_dict(best_model["state_dict"])

    all_files = pd.read_csv(config.train_csv)
    all_gen = HumanDataset(all_files, config.train_dir, augument=False)
    all_loader = DataLoader(all_gen, 1, shuffle=False, pin_memory=True, num_workers=4)

    losses = AverageMeter()
    f1 = F1Meter()
    model.cuda()
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(all_loader)):
            images_var = images.cuda(non_blocking=True)
            target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
            output = model(images_var)
            # f1.update(output.sigmoid().cpu() > thresholds, target)
            f1.update(output.sigmoid().cpu() > torch.from_numpy(thresholds).float(), target)
        print('final loss: %.4f\nfinal f1: %.4f\n' % (losses.avg, f1.f1))


def main(args):
    # mkdir
    if not os.path.exists(config.logs_dir):
        os.mkdir(config.logs_dir)
    if not os.path.exists(config.weights):
        os.mkdir(config.weights)
    if not os.path.exists(config.submit):
        os.makedirs(config.submit)
    if not os.path.exists(config.weights + config.model_name + os.sep + str(args.fold)):
        os.makedirs(config.weights + config.model_name + os.sep + str(args.fold))
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)

    # get model
    model = get_net()
    model.cuda()

    if args.mode == 'train':
        training(model, args.fold, args)
    elif args.mode == 'test':
        testing(model, args.fold, args)
    elif args.mode == 'evaluate':
        evaluating(model, args.fold, args)
    else:
        raise ValueError('Unknown Mode {}'.format(args.mode))


def _parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # base configs
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--fold', type=int, default=0)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(_parse_arguments(sys.argv[1:]))
