#!/usr/bin/python3
'''
    Ejercicio de deteccion de rostros
    Autor: Juan Felipe Jaramillo Hernandez
'''
import cv2
import PIL

import argparse

import matplotlib.pyplot as plt

from multiprocessing.pool import ThreadPool

import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import datetime


class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, (3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), stride=1, padding=1)

        self.conv3 = nn.Conv2d(64, 128, (3, 3), stride=1, padding=1)
        self.fc1 = nn.Linear(1152, 2)

        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.dropout2d = nn.Dropout2d(0.15)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x, silent=True):
        
        silent or print(f"x: {x.shape}")
        
        x = self.conv1(x)
        x = self.dropout2d(x)
        x = self.activation(x)
        x = self.maxpool(x)
        silent or print(f"conv1: {x.shape}")
        
        x = self.conv2(x)
        x = self.dropout2d(x)
        x = self.activation(x)
        x = self.maxpool(x)
        silent or print(f"conv2: {x.shape}")
        
        x = self.conv3(x)
        x = self.dropout2d(x)
        x = self.activation(x)
        x = self.maxpool(x)
        silent or print(f"conv3: {x.shape}")
        
   
        x = nn.Flatten()(x)
        silent or print(f"flatten: {x.shape}")
        
        x = self.fc1(x)
        silent or print(f"fc1: {x.shape}")
        
        return x

def ExtractPatches(img, pyramidsizes, size, stride):

    rowcol_patches = []
    unfolded = torch.tensor([], dtype=torch.uint8)
    for pyr in pyramidsizes:
        
        x = cv2.resize(img, (0,0), fx=pyr, fy=pyr, interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(x)
        x = x.permute(2, 0, 1)
        try:
            minipatches = x.unfold(1, size, stride).unfold(2, size, stride)
            _, row, col, _, _ = minipatches.shape
            rowcol_patches.append((row, col))
            minipatches = minipatches.contiguous().view(minipatches.size(0), -1, size, size)
            minipatches = minipatches.permute(1,0,2,3)
            unfolded = torch.cat((unfolded, minipatches),0)
            del minipatches
        except Exception as e:
            #print(e)
            pass
    return unfolded, rowcol_patches

def Predict(classifier, inputs):
        
    inputs = inputs
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        outputs = classifier(inputs)
        out = softmax(outputs)
    return out

def FindBboxes(probs, rowcol_patches, pyramidsizes, size, stride, thr = 0.99):

    row,col = rowcol_patches.pop(0)
    scale = pyramidsizes.pop(0)
    cont = 0
    scores = []
    bboxes = []
    for i in range(len(probs)):
        if probs[i][0] > thr:
            patchrow = (cont) // col
            patchcol = (cont) % col

            coordh = int((patchrow*stride) * (1.0/scale))
            coordw = int((patchcol*stride) * (1.0/scale))
            boxsize = int(size * (1.0/scale))
            
            bboxes.append((coordw, coordh, boxsize, boxsize))
            scores.append(probs[i][0])

        if cont == row*col-1:
            try:
                row,col = rowcol_patches.pop(0)
                scale = pyramidsizes.pop(0)
            except IndexError as e:
                pass
            cont = 0
        else:
            cont += 1
    return bboxes, scores

def nms(bboxes, scores, threshold):

    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    suppressed_indices = []
    for i in range(len(sorted_indices)):
        idx = sorted_indices[i]
        if idx not in suppressed_indices:
            for j in range(i+1, len(sorted_indices)):
                jdx = sorted_indices[j]
                if jdx not in suppressed_indices and jdx != idx:
                    if iou(bboxes[idx], bboxes[sorted_indices[j]]) > threshold:
                        suppressed_indices.append(sorted_indices[j])
    return [i for i in range(len(scores)) if i not in suppressed_indices]

def iou(box1, box2):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)
    interArea = abs(max((xB - xA + 1),0) * max((yB - yA + 1),0))
    box1Area = w1 * h1 + 1
    box2Area = w2 * h2 + 1
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou


if __name__ == '__main__':

    start_time = datetime.datetime.now()



    parser = argparse.ArgumentParser(
                        prog='FaceDetector',
                        description='Programa para detectar rostros',
                        epilog='Text at the bottom of help')
    parser.add_argument('imagefile', metavar='file', type=str,
                        help='ruta de la imagen a procesar')
    parser.add_argument('--size', type=int, default=0, metavar='int',
                        help='tamaño del patch (default: 33')
    parser.add_argument('--stride', type=int, default=0, metavar='int',
                        help='espacio entre patches (default: 3')
    parser.add_argument('--thr', type=float, default=0.996, metavar='float',
                        help='umbral de probabilidad de cara (default: 0.996)')
    parser.add_argument('--no-silent', action='store_false', default=True,
                        help='activa la verbosidad del programa')   
    args = parser.parse_args()



    patch_size = args.size
    patch_stride = args.stride
    thr = args.thr
    silent = args.no_silent

    if thr > 1.0:
        thr = 0.996
        print(f"umbral de probabilidad de cara establecido en {thr}")

    patch_height = 33
    patch_width = 33

    pyramidsizes = [0.8, 0.6, 0.4, 0.2]

    device = 'cpu'
    silent or print(f"device: {device}")


    classifier = FaceNet()
    classifier.load_state_dict(torch.load('facedetector.pt', map_location=torch.device('cpu')))
    classifier = classifier.to(device)
    classifier.eval()


    img = PIL.Image.open(args.imagefile)
    img = np.array(img)


    if patch_size == 0 or patch_stride == 0:
        imgsize = img.shape

        if imgsize[1]/imgsize[0] > 1.0:
            imgminsize = imgsize[0]
        else: 
            imgminsize = imgsize[1]

        patch_size = int ( imgminsize * pyramidsizes[-1] * 0.85 )
        patch_stride = int ( patch_size / 10 )

        if patch_size < 33:
            patch_size = 33
        if patch_stride < 3:
            patch_stride = 3

    silent or print(f"patch size: {patch_size}")
    silent or print(f"patch stride: {patch_stride}")
    

    unfolded, rowcol_patches = ExtractPatches(img, pyramidsizes, patch_size, patch_stride)
    silent or print(f"extracted patches size: {unfolded.shape}")


    data_transforms = transforms.Compose([
            transforms.Resize((patch_height,patch_width), antialias=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    n = len(unfolded)
    if n>30000:
        batch_size = 30000
        r = int(n / batch_size) + (n % batch_size > 0)


        probs = torch.tensor([])
        for bid in range(r):

            batch = unfolded[bid*batch_size:(bid+1)*batch_size]
            batch = data_transforms(batch.float()/255.).to(device)

            prediction = Predict(classifier, batch)
            probs = torch.cat((probs, prediction.cpu()),0)
    else:
        batch = data_transforms(unfolded.float()/255.).to(device)
        probs = Predict(classifier, batch)  
    
    del unfolded
    silent or print(probs.shape)

    bboxes, scores = FindBboxes(probs, rowcol_patches, pyramidsizes, patch_size, patch_stride, thr=thr)
    del probs
    
    surviving_indices = nms(bboxes, scores, 0.01)

    dimg = img.copy()
    for idx in surviving_indices:
        x, y, w, h = bboxes[idx]
        dimg = cv2.rectangle(dimg,(x, y),(x+w,y+h),(0,255,0),3)
        cv2.putText(dimg, str(np.round(scores[idx].numpy(),3)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, 2)
                    
    dimg = cv2.cvtColor(dimg, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"Faces_{args.imagefile}", dimg)
    print(f"out: Faces_{args.imagefile}")

    n2 = datetime.datetime.now()
    elp = n2 - start_time    
    print(f'--- elapsed time: {elp.total_seconds()} seconds ---')
