# Copyright (c) 2019 Microsoft
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# To prep each node, we'll create symlinks (based on CosmosDB information) within a folder structure that Pytorch is looking for
# for the data loader.
import os
from itertools import product

import pydocumentdb.document_client as document_client
from sklearn.model_selection import train_test_split

BASE_FOLDER = '/tmp/photos/{folder}/'

CHAR_CONVERSION = {'>': "_gt_",
                   ' ': "_"}

COSMOS_SETTINGS = {"COSMOSDB_ENDPOINT": None,
                   "COSMOSDB_KEY": None,
                   "COSMOSDB_DB": None,
                   "COSMOSDB_COLL": None}


def new_folder_name(folder):
    """ Create new folder names - by using the CHAR_CONVERSION to replace characters that may cause problems during
    data reading, etc."""

    for old, new in CHAR_CONVERSION.items():
        folder = folder.replace(old, new)

    return folder


# Query CosmosDB and get all applicable images
def create_client():
    """Create a new DocumentDB Client"""

    return document_client.DocumentClient(url_connection=COSMOS_SETTINGS['COSMOSDB_ENDPOINT'],
                                          auth={'masterKey': COSMOS_SETTINGS['COSMOSDB_KEY']})


def create_folders(client, coll_link, test_set=False):
    """Create new folders where the symbolic links will be added"""

    query = 'SELECT DISTINCT c.metadata.woundHealTime FROM c WHERE c.metadata.study = "Explore"'
    folders = list(map(lambda x: x['woundHealTime'], client.QueryDocuments(coll_link, query=query)))

    if test_set:
        new_folders = list(map(lambda x: '/'.join(x), product(['train', 'test'], folders)))
    else:
        new_folders = folders

    for folder in new_folders:
        folder = new_folder_name(folder)

        os.makedirs(BASE_FOLDER.format(folder=folder), exist_ok=True)


# Loop through images and specified label field and create folders
def create_symlinks(client, coll_link, source_directory, dest_directory, test_set=False, random_seed=None,
                    train_size=0.7):
    """Create the symbolic links
    :param source_directory:
    """

    query = 'SELECT * FROM c WHERE c.metadata.study = "Explore"'
    images = list(map(lambda x: (x['filePath'].split('/')[-1],
                                 new_folder_name(x['metadata']['woundHealTime'])),
                      client.QueryDocuments(coll_link, query=query)))

    if test_set:
        image_sets = {}
        image_sets['train'], image_sets['test'] = train_test_split(images, train_size=train_size, stratify=list(
            map(lambda x: x[1], images)), random_state=random_seed)

        new_images = []
        for label, dataset in image_sets.items():
            new_images += list(map(lambda x: (x[0], label + "/" + x[1]), dataset))

        images = new_images

    for image in images:
        os.symlink(src=os.path.join(source_directory, image[0]),
                   dst=os.path.join(BASE_FOLDER.format(folder=image[1]) + image[0]))


def prepare_node(src_dir, dest_dir, test_set=False, random_seed=None, train_size=0.7):
    """ Orchestration function to take the steps necessary to prepare each node for training """
    db_link = "dbs/{0}".format(COSMOS_SETTINGS['COSMOSDB_DB'])
    coll_link = db_link + "/colls/{0}".format(COSMOS_SETTINGS['COSMOSDB_COLL'])

    dest_dir = os.path.join(dest_dir, "{folder}")

    client = create_client()

    # Create the new folders
    create_folders(client=client, coll_link=coll_link, test_set=test_set)

    # Create symbolic links between the mounted blob store and a folder structure
    # this is done to make it easy to create a reader for Pytorch
    create_symlinks(client=client, coll_link=coll_link, source_directory=src_dir, dest_directory=dest_dir,
                    test_set=test_set, random_seed=random_seed, train_size=train_size)
