# Copyright (c) 2019 Microsoft
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torchvision.models as models
from .classifiers import densenet_classifier, resnet_and_inception_classifier, \
    squeezenet_classifier, vgg_classifier, alexnet_classifier

MODELS_DICT = {
    'alexnet': {'model': models.alexnet, 'family': 'alexnet'},
    'densenet121': {'model': models.densenet121, 'family': 'densenet'},
    'densenet161': {'model': models.densenet161, 'family': 'densenet'},
    'densenet169': {'model': models.densenet169, 'family': 'densenet'},
    'densenet201': {'model': models.densenet201, 'family': 'densenet'},
    'inception_v3': {'model': models.inception_v3, 'family': 'inception'},
    'resnet101': {'model': models.resnet101, 'family': 'resnet'},
    'resnet152': {'model': models.resnet152, 'family': 'resnet'},
    'resnet18': {'model': models.resnet18, 'family': 'resnet'},
    'resnet34': {'model': models.resnet34, 'family': 'resnet'},
    'resnet50': {'model': models.resnet50, 'family': 'resnet'},
    'squeezenet1_0': {'model': models.squeezenet1_0, 'family': 'squeezenet'},
    'squeezenet1_1': {'model': models.squeezenet1_1, 'family': 'squeezenet'},
    'vgg11': {'model': models.vgg11, 'family': 'vgg'},
    'vgg11_bn': {'model': models.vgg11_bn, 'family': 'vgg'},
    'vgg13': {'model': models.vgg13, 'family': 'vgg'},
    'vgg13_bn': {'model': models.vgg13_bn, 'family': 'vgg'},
    'vgg16': {'model': models.vgg16, 'family': 'vgg'},
    'vgg16_bn': {'model': models.vgg16_bn, 'family': 'vgg'},
    'vgg19': {'model': models.vgg19, 'family': 'vgg'},
    'vgg19_bn': {'model': models.vgg19_bn, 'family': 'vgg'}
}

FAMILY_CLASSIFIER = {
    'alexnet': alexnet_classifier,
    'densenet': densenet_classifier,
    'inception': resnet_and_inception_classifier,
    'resnet': resnet_and_inception_classifier,
    'squeezenet': squeezenet_classifier,
    'vgg': vgg_classifier
}


class VisionModelFactory(object):

    def __init__(self):
        pass

    @staticmethod
    def create_model(model_architecture, num_classes, freeze_layers=True, classification_layer=None):
        model_obj = MODELS_DICT[model_architecture]

        print('Downloading {0} pretrained model'.format(model_architecture))
        model = model_obj['model'](pretrained=True)

        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False

        # Call the appropriate function to replace the
        model = FAMILY_CLASSIFIER[model_obj['family']](model, num_classes, classification_layer)

        return model

    @staticmethod
    def unlock_model(model, num_layers=None):
        for param in model.parameters():
            param.requires_grad = True