
import torch
import torch.nn as nn
from torch.autograd import Variable
from convex_adversarial import robust_loss, robust_loss_parallel
import torch.optim as optim

import numpy as np
import time
import gc

from trainer import *
import cv2 as cv
import random

def train_ibp(loader, model, opt, epsilon, epoch, log, verbose):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    model.train()

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
        data_time.update(time.time() - end)


        alpha = 0.5
        out = model(Variable(X))
        out_l, out_h = model.forward2(Variable(X - epsilon), Variable(X + epsilon))
        out_hat = out_h
        for i in range(out_l.shape[0]):
            out_hat[i][y[i]] = out_l[i][y[i]]
        ce = alpha * nn.CrossEntropyLoss()(out, Variable(y)) + (1.0-alpha) * nn.CrossEntropyLoss()(out_hat, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        opt.zero_grad()
        ce.backward()
        opt.step()

        batch_time.update(time.time()-end)
        end = time.time()
        losses.update(ce.data.item(), X.size(0))
        errors.update(err, X.size(0))

        print(epoch, i, ce.data.item(), err, file=log)
        if verbose and i % verbose == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, errors=errors))
        log.flush()


def evaluate_rotations(loader, model, epsilon, epoch, log, verbose):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    model.eval()

    end = time.time()
    for i, (X,y) in enumerate(loader):
        # print("Got value of X")
        # print(X.numpy()[0][0])

        npX = np.array(X)
        cols = 28
        rows = 28
        for i in range(npX.shape[0]):
            rotation_degree = random.randint(-15, 15)
            rotation_degree = rotation_degree + 15 * np.sign(rotation_degree)
            # print("Random", rotation_degree)
            M = cv.getRotationMatrix2D((cols / 2, rows / 2), rotation_degree, 1)
            npX[i][0] = cv.warpAffine(npX[i][0], M, (cols, rows))
            # cv.namedWindow('image', cv.WINDOW_NORMAL)
            # cv.imshow('image', npX[i][0])
            # cv.resizeWindow('image', 600, 600)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
        X = torch.from_numpy(npX)
        X,y = X.cuda(), y.cuda()
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        # print to logfile
        print(epoch, i, ce.data.item(), err, file=log)

        # measure accuracy and record loss
        losses.update(ce.data.item(), X.size(0))
        errors.update(err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and i % verbose == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {error.val:.3f} ({error.avg:.3f})'.format(
                      i, len(loader), batch_time=batch_time, loss=losses,
                      error=errors))
        log.flush()

    print(' * Error {error.avg:.3f}'
          .format(error=errors))
    return errors.avg
