# Copyright (c) 2019 Microsoft
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

#
#
# Resnet : fc (linear)
#
# Densenet : classifier (linear)
#
# Squeezenet : classifier
#   Sequential(
#   (0): Dropout(p=0.5)
#   (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
#   (2): ReLU(inplace)
#   (3): AvgPool2d(kernel_size=13, stride=1, padding=0)
# )
#
# Inception V3 : fc (linear)
#
# Alexnet : classifier
#   Sequential(
#   (0): Dropout(p=0.5)
#   (1): Linear(in_features=9216, out_features=4096, bias=True)
#   (2): ReLU(inplace)
#   (3): Dropout(p=0.5)
#   (4): Linear(in_features=4096, out_features=4096, bias=True)
#   (5): ReLU(inplace)
#   (6): Linear(in_features=4096, out_features=1000, bias=True)
# )
#
# VGG : classifier
#   Sequential(
#   (0): Linear(in_features=25088, out_features=4096, bias=True)
#   (1): ReLU(inplace)
#   (2): Dropout(p=0.5)
#   (3): Linear(in_features=4096, out_features=4096, bias=True)
#   (4): ReLU(inplace)
#   (5): Dropout(p=0.5)
#   (6): Linear(in_features=4096, out_features=1000, bias=True)
# )

from .visionmodelfactory import VisionModelFactory