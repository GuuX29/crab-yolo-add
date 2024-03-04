#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2024/2/23 19:51
# @Author: GuuX
# @File  : efn_eval_new.py.py

import os
import pandas as pd
import time
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils import loss, auto_augment, dataset, mymodel, tools, args, ranger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def validate(val_loader, model, criterion):
    model.eval()
    device = next(model.parameters()).device
    losses = []
    predictions = []
    targets = []

    with torch.no_grad():
        for input, target in val_loader:
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = criterion(output, target)
            losses.append(loss.item())
            predictions.append(output.argmax(dim=1))
            targets.append(target)

    losses = torch.tensor(losses).mean().item()
    predictions = torch.cat(predictions).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='macro')
    conf_matrix = confusion_matrix(targets, predictions)

    def plot_confusion_matrix(conf_matrix):
        plt.figure(figsize=(10, 8))
        sns.set(font_scale=1.2)
        ax = sns.heatmap(conf_matrix, annot=True, fmt=".1f", cmap="Blues", cbar=False)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig('conf_matrix.png')
        plt.show()

    # Call the plot_confusion_matrix function after generating the confusion matrix
    conf_matrix = confusion_matrix(targets, predictions)
    plot_confusion_matrix(conf_matrix)

    unique_labels = list(range(conf_matrix.shape[0]))
    df = pd.DataFrame(conf_matrix, columns=unique_labels, index=unique_labels)
    df.to_csv('conf_matrix.csv')

    return losses, precision, recall, f1

# Call the validate function in the main code
def main():
    train_df = pd.read_csv('./images/train.csv')
    train_df["FileID"] = "images/train/" + train_df["SpeciesID"].astype(str) + "/" + train_df["FileID"] + '.jpg'

    skf = StratifiedKFold(n_splits=5, random_state=409, shuffle=True)

    train_batch_size = args.train_args.batch_size
    size = args.train_args.image_size
    epochs = 1
    using_cutmix = False
    using_label_smooth = False
    model_path = args.train_args.model_path
    classes = args.train_args.num_class

    if using_label_smooth:
        criterion = loss.CrossEntropyLabelSmooth(classes, epsilon=0.1)
    else:
        criterion = nn.CrossEntropyLoss()

    val_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df['FileID'], train_df['SpeciesID'])):
        val_data = train_df.iloc[val_idx]
        val_loader = dataset.DataLoaderX(
            dataset.QRDataset(list(val_data['FileID']), list(val_data['SpeciesID']), val_transform),
            batch_size=train_batch_size,
            shuffle=False,
            pin_memory=True)

        model = mymodel.Net_b5(num_class=classes)
        if model_path:
            model.load_state_dict(torch.load(model_path))
            print('Loaded model from: {}'.format(model_path))
        model = model.cuda()

        for epoch in range(epochs):
            print('Epoch: ', epoch)
            val_loss, val_precision, val_recall, val_f1 = validate(val_loader, model, criterion)
            print(
                'Validation Loss: {:4f}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(val_loss, val_precision,
                                                                                               val_recall, val_f1))
            break  # Remove this line if you want to run for all epochs

if __name__ == "__main__":
    main()
