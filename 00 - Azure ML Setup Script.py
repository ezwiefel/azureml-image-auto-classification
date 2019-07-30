# Copyright (c) 2019 Microsoft
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

#%% [markdown]

## Create Azure Machine Learning workspace
#### _One Time Only Notebook_
# The AML Workspace stores infomation useful for building our Machine Learning and Data Science models - such as experiment tracking, model management, data stores and other useful artifacts.
#
# ![AML Concepts](https://docs.microsoft.com/en-us/azure/machine-learning/service/media/concept-azure-machine-learning-architecture/taxonomy.png)

#%%

from azureml.core import Workspace, Datastore
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.authentication import InteractiveLoginAuthentication

#%%

# Give details of where the service should be created and what the name should be. This only needs to be done once - all projects can exist within the same workspace.
STORAGE_ACCT_KEY = None
STORAGE_ACCT_NAME = None

SUB_ID = None
RESOURCE_GROUP = None
WORKSPACE_NAME = None
WORKSPACE_REGION = None



# This script will load an already existing workspace
ws = Workspace(workspace_name = WORKSPACE_NAME,
               subscription_id = SUB_ID,
               resource_group = RESOURCE_GROUP                                                                                                  
              )

# This code would create a new workspace
# ws = Workspace.create(workspace_name = workspace_name,
#                       subscription_id = subscription_id,
#                       resource_group = resource_group,
#                       location=workspace_region)

# Save the configuration file for the workspace to DBFS
ws.write_config()

#%% [markdown]

### Create Datastore in Workspace
# First, we'll register the datastore in the workspace. This is a one-time only event.

#%%

ds = Datastore.register_azure_blob_container(workspace=ws, datastore_name="images", 
                                             container_name="images", account_name=STORAGE_ACCT_NAME, 
                                             account_key=STORAGE_ACCT_KEY)

#%% [markdown]

### Create AML Compute Cluster
# Next, we'll create an autoscaling AML Compute layer cluster - with 0 node minimum and 10 mode maximum. We'll create it in WestUS2 - which is the same region that our data is stored in.

#%%

aml_cluster_name = 'gpu-cluster'
provisioning_config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_NC6", # NC6 is GPU-enabled
                                                            autoscale_enabled = True,
                                                            min_nodes = 0, 
                                                            max_nodes = 10,
                                                            description="A GPU enabled cluster.")

# create the cluster
compute_target = ComputeTarget.create(ws, aml_cluster_name, provisioning_config)