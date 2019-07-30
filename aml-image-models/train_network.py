# Copyright (c) 2019 Microsoft
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import argparse
import os

import torch

from prep_node import prepare_node
from network import VisionModelFactory
from load_data import create_datasets
from azureml.core.run import Run


def train_model(runner, model, data_loaders,
                checkpoint_folder,
                criterion, optimizer, scheduler,
                total_epochs=None, previous_epochs=0, num_epochs=25):

    if not total_epochs:
        total_epochs = num_epochs

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1+previous_epochs, total_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            total = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                total += labels.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels.data).sum().item()

            # Log metrics and save checkpoint file
            runner.log('Accuracy - {0}'.format(phase), running_corrects / total)
            runner.log('Loss - {0}'.format(phase), running_loss / total)
            print('Accuracy - {0}'.format(phase), running_corrects / total)
            print('Loss - {0}'.format(phase), running_loss / total)

        # Save a checkpoint after every 5 epochs
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_folder, 'checkpoint-epoch-{0}.pth'.format(epoch)))

    return model


def prep_training_objects(model, learning_rate, momentum, step_size, gamma, optimizer_type='sgd'):
    criterion = torch.nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    if optimizer_type.upper() == 'SGD':
        optimizer = torch.optim.SGD([param for param in model.parameters() if param.requires_grad],
                                    lr=learning_rate,
                                    momentum=momentum)
    elif optimizer_type.upper() == 'ADAM':
        optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad],
                                    lr=learning_rate)
    else:
        raise NotImplementedError('Optimizer {0} is not recognized or not implemented.')

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return criterion, optimizer, scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/outputs", type=str, help="Directory with Images to Train On")
    parser.add_argument('--output-dir', default='./outputs', type=str, help="Output directory to write checkpoints to")
    parser.add_argument('--logs-dir', default="/logs", type=str, help="Directory to write logs to")
    parser.add_argument('--num-epochs', default=2, type=int, help="The number of epochs to train the model for")
    parser.add_argument('--network-name', default='resnet50', type=str,
                        help="The name of the pretrained model to be used. Must be in torchvision.models")
    parser.add_argument('--minibatch-size', default=32, type=int,
                        help="The number of images to be included in each minibatch")
    parser.add_argument('--do-not-shuffle-photos', action='store_false', help='Do not shuffle the photos at each epoch')
    parser.add_argument('--num-dataload-workers', default=4, type=int,
                        help='The number of workers to load the pictures')
    parser.add_argument('--learning-rate', default=0.001, type=float, help="The learning rate to use while training the neural network")
    parser.add_argument('--momentum', default=0.9, type=float, help='The momentum to use while training the neural network')
    parser.add_argument('--step-size', default=7, type=int, help='The step size for the learning rate scheduler - '
                                                                 'will reduce learning rate every x epochs')
    parser.add_argument('--gamma', default=0.9, type=float, help='The gamma setting for the learning rate scheduler')
    parser.add_argument('--optimizer-type', default='sgd', type=str, help='The optimizer algorithm to use. '
                                                                          'Currenly SGD and Adam are supported')
    parser.add_argument('--epochs-before-unfreeze-all', default=0, type=int, help='The number of epochs to train before '
                                                                                  'unfreezing all layers of the model')
    parser.add_argument('--checkpoint-epochs', default=25, type=int, help='How often a checkpoint file is saved')
    args = parser.parse_args()

    dest_dir = '/tmp/photos/'

    # Prep the data and dataloaders
    prepare_node(src_dir=args.data_dir, dest_dir=dest_dir, test_set=True, random_seed=57522, train_size=0.8)
    image_datasets = create_datasets(dest_dir)

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.minibatch_size,
                                                  shuffle=args.do_not_shuffle_photos,
                                                  num_workers=args.num_dataload_workers)
                   for x in ['train', 'test']}

    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = VisionModelFactory.create_model(args.network_name, len(class_names))
    model.to(device)

    # Log the details
    runner = Run.get_context()

    runner.log("Network Architecture", args.network_name)
    runner.log("Minibatch Size", args.minibatch_size)
    runner.log("Number of Epochs", args.num_epochs)
    runner.log("Shuffle Photos", args.do_not_shuffle_photos)
    runner.log("Number of Dataloaders", args.num_dataload_workers)
    runner.log("Learning Rate", args.learning_rate)
    runner.log("Momentum", args.momentum)
    runner.log('Step Size', args.step_size)
    runner.log('Gamma', args.gamma)
    runner.log('Optimizer', args.optimizer_type)
    runner.log('Epochs before Unfreeze Model', args.epochs_before_unfreeze_all)


    if args.epochs_before_unfreeze_all != 0:
        # Create criterion, optimizer and LR scheduler
        criterion, optimizer, scheduler = prep_training_objects(model, args.learning_rate, args.momentum,
                                                                args.step_size, args.gamma)

        model = train_model(runner=runner, model=model, data_loaders=dataloaders, checkpoint_folder=args.output_dir,
                            criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                            num_epochs=args.epochs_before_unfreeze_all, total_epochs=args.num_epochs, previous_epochs=0)
        print()
        print("================")
        print("Unfreezing Model")
        print("================")
        VisionModelFactory.unlock_model(model)
        previous_epochs = args.epochs_before_unfreeze_all
        num_epochs = args.num_epochs - previous_epochs

    else:
        num_epochs = args.num_epochs
        previous_epochs = 0

    # Create criterion, optimizer and LR scheduler
    criterion, optimizer, scheduler = prep_training_objects(model, args.learning_rate, args.momentum,
                                                            args.step_size, args.gamma)

    model = train_model(runner=runner, model=model, data_loaders=dataloaders, checkpoint_folder=args.output_dir,
                        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                        num_epochs=num_epochs, total_epochs=args.num_epochs, previous_epochs=previous_epochs)



    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))