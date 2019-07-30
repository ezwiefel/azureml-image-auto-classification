# Copyright (c) 2019 Microsoft
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch.nn as nn


def densenet_classifier(model, num_classes, classification_layer=None):
    num_ftrs = model.classifier.in_features

    if classification_layer:
        model.classifier = classification_layer
    else:
        model.classifier = nn.Linear(in_features=num_ftrs, out_features=num_classes)

    return model


def resnet_and_inception_classifier(model, num_classes, classification_layer=None):
    num_ftrs = model.fc.in_features

    if classification_layer:
        model.fc = classification_layer
    else:
        model.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes)

    return model


def alexnet_classifier(model, num_classes, classification_layer=None):
    num_ftrs = model.classifier[1].in_features

    # Recreate classification layer as specified by Pytorch model zoo
    if not classification_layer:
        classification_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=num_ftrs, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        )

    model.classifier = classification_layer
    return model

def squeezenet_classifier(model, num_classes, classification_layer=None):
    raise NotImplementedError("squeezenet architectures have not been implemented for these vision models.")

def vgg_classifier(model, num_classes, classification_layer=None):
    num_ftrs = model.classifier[0].in_features

    if not classification_layer:
        classification_layer = nn.Sequential(
          nn.Linear(in_features=num_ftrs, out_features=4096, bias=True),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.5),
          nn.Linear(in_features=4096, out_features=4096, bias=True),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.5),
          nn.Linear(in_features=4096, out_features=num_classes, bias=True),
        )

    model.classifier = classification_layer
    return model