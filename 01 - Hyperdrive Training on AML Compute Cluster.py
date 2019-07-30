# Copyright (c) 2019 Microsoft
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

#%% [markdown]
## Training Pytorch using Hyperdrive on AML Compute

#%%
from azureml.core import Workspace, Datastore
from azureml.core.compute import ComputeTarget
from azureml.core.run import Run
from azureml.train.estimator import *
from azureml.train.dnn import PyTorch
import azureml.train.hyperdrive as hd
from azureml.core.authentication import InteractiveLoginAuthentication
import pandas as pd

#%% 
# Load workspace information from configuration file
ws = Workspace.from_config()

# Get datastore and compute information
ds = ws.datastores['images']
ct = ws.compute_targets['gpu-cluster']

#%% [markdown]

#### Submit Hyperdrive Run
# We've written the `train_network.py` script to accept parameters - the possible parameters are:
    
# | Parameter                      | Default Value | Description|
# |--------------------------------|---------------|------------|
# | `--data-dir`                   | `/outputs`    | Directory with Images to Train On |
# | `--output-dir`                 | `/outputs`    | Output directory to write checkpoints to |
# | `--logs-dir`                   | `/logs`       | Directory to write logs to |
# | `--num-epochs`                 | 2             | The number of epochs to train the model for |
# | `--minibatch-size`             | 32            | The number of images to be included in each minibatch |
# | `--do-not-shuffle-photos`      | False         | Do not shuffle the photos at each epoch |
# | `--num-dataload-workers`       | 4             | The number of workers to load the pictures |
# | `--learning-rate`              | 0.001         | The learning rate to use while training the neural network |
# | `--momentum`                   | 0.9           | The momentum to use while training the neural network |
# | `--step-size`                  | 7             | The momentum to use while training the neural network |
# | `--momentum`                   | 0.9           | The step size for the learning rate scheduler - will reduce learning rate every _x_ epochs |
# | `--gamma`                      | 0.9           | The gamma setting for the learning rate scheduler |
# | `--optimizer-type`             | `sgd`         | The optimizer algorithm to use. Currenly SGD and Adam are supported |
# | `--epochs-before-unfreeze-all` | 0             | The number of epochs to train before unfreezing all layers of the model |
# | `--checkpoint-epochs`          | 25            | How often a checkpoint file is saved
# | `--network-name` | `resnet50`    | The name of the pretrained model to be used. <br> Possible values are: <br> `alexnet` <br> `densenet121` <br> `densenet161` <br> `densenet169` <br> `densenet201` <br> `resnet101` <br> `resnet152` <br> `resnet18` <br> `resnet34` <br> `resnet50` <br> `vgg11` <br> `vgg11_bn` <br> `vgg13` <br> `vgg13_bn` <br> `vgg16` <br> `vgg16_bn` <br> `vgg19` <br> `vgg19_bn`  |

#%% [markdown]

###### Create PyTorch estimator object
# This is used to submit the Pytorch job and will be an input for the HyperDrive run.

#%%

# Set the static script parameters
script_params={
    '--data-dir': ds.as_mount(),
    '--num-epochs': 200, # This number would likely be increased as we move to production
    '--momentum': 0.9,
    '--num-dataload-workers': 6,
    '--epochs-before-unfreeze-all': '0', # Don't unfreeze the model - since the performance degrades based on the number of images we have in the test set
}

conda_packages=['pytorch', 'scikit-learn']
pip_packages=['pydocumentdb', 'torchvision']

#%%

estimator = PyTorch(source_directory='./aml-image-models',
                      compute_target=ct,
                      entry_script='train_network.py',
                      script_params=script_params,
                      node_count=1,
                      process_count_per_node=1,
                      conda_packages=conda_packages,
                      pip_packages=pip_packages,
                      use_gpu=True)

#%%

# Create Experiment object - this will be used to submit the Hyperdrive run and store all the given parameters
experiment_hd = Experiment(workspace=ws, name='hyperdrive')

#%% [markdown]

###### Create Random Parameter Sampler

#%%

# Parameter space to sweep over - uses Random Parameter Sampling
ps = hd.RandomParameterSampling(
    {
        '--network-name': hd.choice('densenet201', 'resnet152', 'resnet34', 'alexnet', 'vgg19_bn'),
        '--minibatch-size': hd.choice(8, 16),
        '--learning-rate': hd.uniform(0.00001, 0.001),
        '--step-size': hd.choice(10, 25, 50),  # How often should the learning rate decay
        '--gamma': hd.uniform(0.7, 0.99),      # The decay applied to the learning rate every {step-size} steps
        '--optimizer-type': hd.choice('sgd', 'adam')
    }
)

#%% [markdown]

###### Create Early Termination Policy

#%%

# check every 10 iterations and if the primary metric (epoch_val_acc) falls
# outside of the range of 20% of the best recorded run so far, terminate it.
etp = hd.BanditPolicy(slack_factor = 0.2, evaluation_interval = 10, delay_evaluation=15)

# Another termination policy - cancel the lowest performing 25% of runs every 15 epochs
# etp = hd.TruncationSelectionPolicy(truncation_percentage=25, evaluation_interval=15)

#%% [markdown]

###### Create HyperDrive Run Configuration

#%%

# Hyperdrive run configuration
hrc = hd.HyperDriveRunConfig(
    estimator = estimator,
    hyperparameter_sampling = ps,
    policy = etp,
    # metric to watch (for early termination)
    primary_metric_name = 'Loss - test',
    # terminate if metric falls below threshold
    primary_metric_goal = hd.PrimaryMetricGoal.MINIMIZE,
    max_total_runs = 3,
    max_concurrent_runs = 3,
)

#%% [markdown]

###### Submit Hyperdrive run configuration to `experiment_hd.submit()` function

#%%

# Submit the run
hd_run = experiment_hd.submit(hrc)
print(hd_run.get_portal_url())

#%%

best_run = hd_run.get_best_run_by_primary_metric()
print(best_run.get_portal_url())

#%%

# Either Download the saved model file to DBFS...
best_run.download_file("outputs/final_model.pth", "saved_model/final_model.pth")

#%% [markdown]

###### Get Hyperparameters from Best Run
# To get the hyperparameters from the best run, we can use the GUI or we can programatically retrive them.

#%%

params = best_run.get_details()['runDefinition']['Arguments']

#%%

# ... or register the model with the AML Model Management Service
model = best_run.register_model(model_name='Training_Model', model_path='outputs/final_model.pth')

#%%

models = [(mdl.name, mdl.version) for mdl in model.list(ws)]
print("Version\t\tModel Name")
for mdl in models:
  print(mdl[1], "\t\t", mdl[0])
