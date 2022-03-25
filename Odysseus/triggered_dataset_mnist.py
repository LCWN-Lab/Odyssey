
#!/usr/bin/env python3

import argparse
import glob
import logging.config
import multiprocessing
import os
import time
import sys
from numpy.random import RandomState


import torch
import trojai.modelgen.architecture_factory as tpm_af
import trojai.modelgen.architectures.mnist_architectures as cfa
import trojai.modelgen.config as tpmc
import trojai.modelgen.data_manager as tpm_tdm
import trojai.modelgen.model_generator as mg

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, os.path.abspath('./datagen/'))
import mnist
from mnist_utils import download_and_extract_mnist_file, convert

import trojai.datagen.insert_merges as tdi
import trojai.datagen.datatype_xforms as tdd
import trojai.datagen.image_triggers as tdt
import trojai.datagen.merge_interface as td_merge
import trojai.datagen.common_label_behaviors as tdb
import trojai.datagen.experiment as tde
import trojai.datagen.config as tdc
import trojai.datagen.xform_merge_pipeline as tdx
import trojai.datagen.instagram_xforms as tinstx
import shutil

logger = logging.getLogger(__name__)


class DummyMerge(td_merge.Merge):
    def do(self, obj1, obj2, random_state_obj):
        pass

def download_mnist(clean_train_path, clean_test_path, temp_dir):
    # setup file system
    train_csv_dir = os.path.dirname(clean_train_path)
    test_csv_dir = os.path.dirname(clean_test_path)
    try:
        os.makedirs(train_csv_dir)
    except IOError:
        pass
    try:
        os.makedirs(test_csv_dir)
    except IOError:
        pass
    try:
        os.makedirs(temp_dir)
    except IOError:
        pass

    # download the 4 datasets
    logger.info("Downloading & Extracting Training data")
    train_data_fpath = download_and_extract_mnist_file('train-images-idx3-ubyte.gz', temp_dir)
    logger.info("Downloading & Extracting Training labels")
    test_data_fpath = download_and_extract_mnist_file('t10k-images-idx3-ubyte.gz', temp_dir)
    logger.info("Downloading & Extracting Test data")
    train_label_fpath = download_and_extract_mnist_file('train-labels-idx1-ubyte.gz', temp_dir)
    logger.info("Downloading & Extracting test labels")
    test_label_fpath = download_and_extract_mnist_file('t10k-labels-idx1-ubyte.gz', temp_dir)

    # convert it to the format we need
    logger.info("Converting Training data & Labels from ubyte to CSV")
    convert(train_data_fpath, train_label_fpath, clean_train_path, 60000, description='mnist_train_convert')
    logger.info("Converting Test data & Labels from ubyte to CSV")
    convert(test_data_fpath, test_label_fpath, clean_test_path, 10000, description='mnist_test_convert')

    logger.info("Cleaning up...")
    os.remove(os.path.join(temp_dir, 'train-images-idx3-ubyte.gz'))
    os.remove(os.path.join(temp_dir, 'train-labels-idx1-ubyte.gz'))
    os.remove(os.path.join(temp_dir, 't10k-images-idx3-ubyte.gz'))
    os.remove(os.path.join(temp_dir, 't10k-labels-idx1-ubyte.gz'))
    os.remove(os.path.join(temp_dir, 'train-images-idx3-ubyte'))
    os.remove(os.path.join(temp_dir, 'train-labels-idx1-ubyte'))
    os.remove(os.path.join(temp_dir, 't10k-images-idx3-ubyte'))
    os.remove(os.path.join(temp_dir, 't10k-labels-idx1-ubyte'))




# xform data to conform to PyTorch
def img_transform(x):
	x = x.permute(2, 0, 1)
	return x

parser = argparse.ArgumentParser(description='MNIST Data & Model Generator and Experiment Iterator')

# Args related to data generation
parser.add_argument('--data_folder', type=str, default ='./data/mnist/', help='Path to folder containing mnist data')
parser.add_argument('--experiment_path', type=str, default='./data/mnist/', help='Root Folder of output')
parser.add_argument('--train', type=str, help='CSV file which contains raw MNIST Training data',
					default='./data/mnist/mnist_clean/train_mnist.csv')
parser.add_argument('--test', type=str, help='CSV file which contains raw MNIST Test data',
					default='./data/mnist/mnist_clean/test_mnist.csv')

# Args related to model generation
parser.add_argument('--log', type=str, help='Log File')
parser.add_argument('--console', action='store_true')
parser.add_argument('--tensorboard_dir', type=str, default=None, help='Folder for logging tensorboard')
parser.add_argument('--gpu', action='store_true', default =True)
parser.add_argument('--early_stopping', action='store_true')


a = parser.parse_args()

# setup data generation
# Setup the files based on user inputs
data_folder = os.path.abspath(a.data_folder)
top_folder = a.experiment_path
train = a.train
test = a.test

# check if the data_folder has the mnist data, if not download it
if not os.path.isdir(data_folder):
	download_mnist(train, test, data_folder)

MASTER_SEED = 1234
master_random_state_object = RandomState(MASTER_SEED)
start_state = master_random_state_object.get_state()

## List of trigger patterns
random_trigger_pattern = ['Triangular90drightPattern','ReverseLambdaPattern', 'RandomPattern', 'RecTriangular90drightPattern',
						  'RecTriangularPattern', 'RecTriangularReversePattern', 'AlphaYPattern', 'AlphaZPattern',
						  'AlphaIPattern', 'AlphaJPattern', 'AlphaKPattern', 'Triangular90dleftPattern', 'AlphaEPattern',
						  'AlphaEReversePattern', 'AlphaAPattern', 'AlphaWPattern', 'AlphaBPattern', 'AlphaLPattern', 
						  'AlphaCPattern', 'AlphaSPattern',  'AlphaDPattern', 'RecTriangular90lefttPattern',
						  'Rec90drightTriangularPattern', 'Rec90dleftTriangularPattern', 'RecTriangular90dleftPattern',
						  'DiamondPattern', 'AlphaHPattern', 'AlphaMPattern', 'AlphaOPattern', 'AlphaQPattern', 'AlphaXPattern', 
						  'AlphaDOPattern', 'RectangularPattern', 'TriangularPattern',  'TriangularPattern47', 'AlphaNPattern',
						  'RectangularPattern_62', 'RandomPattern_62', 'ReverseLambdaPattern_62', 'TriangularReversePattern47', 
						  'OnesidedPyramidReversePattern', 'OnesidedPyramidPattern', 'OnesidedPyramidPattern63'
																					 'AlphaDO1Pattern', 'AlphaDO2Pattern', 'AlphaTPattern']

# Check the type of trigger
for random_trigger_pattern_str in random_trigger_pattern:
	if random_trigger_pattern_str == 'ReverseLambdaPattern':
		trigger_selection = tdt.ReverseLambdaPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'RandomPattern':
		trigger_selection = tdt.RandomRectangularPattern(5, 5, 1, 'channel_assign', {'cval': [234]})
	elif random_trigger_pattern_str == 'RandomPattern_62':
		trigger_selection = tdt.RandomRectangularPattern(6, 2, 1, 'channel_assign', {'cval': [234]})
	elif random_trigger_pattern_str == 'RectangularPattern_62':
		trigger_selection = tdt.RectangularPattern(6, 2, 1, 255)
	elif random_trigger_pattern_str == 'ReverseLambdaPattern_62':
		trigger_selection = tdt.ReverseLambdaPattern(6, 2, 1, 255)
	elif random_trigger_pattern_str == 'RectangularPattern':
		trigger_selection = tdt.RectangularPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'OnesidedPyramidReversePattern':
		trigger_selection = tdt.OnesidedPyramidReversePattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'OnesidedPyramidPattern':
		trigger_selection = tdt.OnesidedPyramidPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'TriangularPattern':
		trigger_selection = tdt.TriangularPattern(3, 5, 1, 255)
	elif random_trigger_pattern_str == 'TriangularPattern47':
		trigger_selection = tdt.TriangularPattern(4, 7, 1, 255)
	elif random_trigger_pattern_str == 'TriangularReversePattern':
		trigger_selection = tdt.TriangularReversePattern(3, 5, 1, 255)
	elif random_trigger_pattern_str == 'TriangularReversePattern47':
		trigger_selection = tdt.TriangularReversePattern(4, 7, 1, 255)
	elif random_trigger_pattern_str == 'OnesidedPyramidPattern63':
		trigger_selection = tdt.OnesidedPyramidPattern(6, 3, 1, 255)
	elif random_trigger_pattern_str == 'Triangular90drightPattern':
		trigger_selection = tdt.Triangular90drightPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'RecTriangular90drightPattern':
		trigger_selection = tdt.RecTriangular90drightPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'Triangular90dleftPattern':
		trigger_selection = tdt.Triangular90dleftPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'RecTriangular90dleftPattern':
		trigger_selection = tdt.RecTriangular90dleftPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'RecTriangularPattern':
		trigger_selection = tdt.RecTriangularPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'RecTriangularReversePattern':
		trigger_selection = tdt.RecTriangularReversePattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'Rec90drightTriangularPattern':
		trigger_selection = tdt.Rec90drightTriangularPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'Rec90dleftTriangularPattern':
		trigger_selection = tdt.Rec90dleftTriangularPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'DiamondPattern':
		trigger_selection = tdt.DiamondPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaEPattern':
		trigger_selection = tdt.AlphaEPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaAPattern':
		trigger_selection = tdt.AlphaAPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaWPattern':
		trigger_selection = tdt.AlphaWPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaBPattern':
		trigger_selection = tdt.AlphaBPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaCPattern':
		trigger_selection = tdt.AlphaCPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaDPattern':
		trigger_selection = tdt.AlphaDPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaEReversePattern':
		trigger_selection = tdt.AlphaEReversePattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaLPattern':
		trigger_selection = tdt.AlphaLPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaPPattern':
		trigger_selection = tdt.AlphaPPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaSPattern':
		trigger_selection = tdt.AlphaSPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaNPattern':
		trigger_selection = tdt.AlphaNPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaTPattern':
		trigger_selection = tdt.AlphaTPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaXPattern':
		trigger_selection = tdt.AlphaXPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaYPattern':
		trigger_selection = tdt.AlphaYPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaZPattern':
		trigger_selection = tdt.AlphaZPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaIPattern':
		trigger_selection = tdt.AlphaIPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaJPattern':
		trigger_selection = tdt.AlphaJPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaKPattern':
		trigger_selection = tdt.AlphaKPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaHPattern':
		trigger_selection = tdt.AlphaHPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaMPattern':
		trigger_selection = tdt.AlphaMPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaOPattern':
		trigger_selection = tdt.AlphaOPattern(5, 4, 1, 255)
	elif random_trigger_pattern_str == 'AlphaQPattern':
		trigger_selection = tdt.AlphaQPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaDOPattern':
		trigger_selection = tdt.AlphaDOPattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaDO1Pattern':
		trigger_selection = tdt.AlphaDO1Pattern(5, 5, 1, 255)
	elif random_trigger_pattern_str == 'AlphaDO2Pattern':
		trigger_selection = tdt.AlphaDO2Pattern(5, 5, 1, 255)

	# Fraction of the total number of samples to be trigerred
	datagen_per_class_trigger_frac = 0.25

	## Define a configuration which inserts a specific pattern at a specified location in the Fashion MNIST image to
	## create a triggered Fashion MNIST dataset--
	# a) For more details on how to configure the Pipeline, check the XFormMergePipelineConfig documentation.
	# b) For more details on any of the objects used to configure the Pipeline, check their respective docstrings.
	one_channel_alpha_trigger_cfg = \
		tdc.XFormMergePipelineConfig(

			# Setup the list of possible triggers that will be inserted into the MNIST data.
			trigger_list = [trigger_selection],

			# Tell the trigger inserter the probability of sampling each type of trigger specified in the trigger
			# list.  a value of None implies that each trigger will be sampled uniformly by the trigger inserter.
			trigger_sampling_prob=None,

			# List any transforms that will occur to the trigger before it gets inserted.  In this case, we do none.
			trigger_xforms=[],

			# List any transforms that will occur to the background image before it gets merged with the trigger.
			# trigger_bg_xforms = [tinstx.GothamFilterXForm()],
			trigger_bg_xforms = [tdd.ToTensorXForm()],

			# List how we merge the trigger and the background.  Because we don't insert a point trigger,
			# the merge is just a no-op
			trigger_bg_merge = tdi.InsertAtRandomLocation('uniform_random_available', tdc.ValidInsertLocationsConfig()),
			# trigger_bg_merge = DummyMerge(),

			# A list of any transformations that we should perform after merging the trigger and the background.
			trigger_bg_merge_xforms=[],

			# Denotes how we merge the trigger with the background.
			merge_type='insert',

			# Specify that all the clean data will be modified.  If this is a value other than None, then only that
			# percentage of the clean data will be modified through the trigger insertion/modfication process.
			per_class_trigger_frac=datagen_per_class_trigger_frac,

			# Specify which classes will be triggered
			triggered_classes = [0,1,2,3,4,5,6,7,8,9]
		)

	logger.info("Generating experiment...")


	## Setup the files based on user inputs ##
	train_csv_file = os.path.abspath(train)
	test_csv_file = os.path.abspath(test)

	train_output_csv_file = 'train_mnist.csv'
	test_output_csv_file = 'test_mnist.csv'

	if not os.path.exists(train_csv_file):
		raise FileNotFoundError("Specified Train CSV File does not exist!")
	if not os.path.exists(test_csv_file):
		raise FileNotFoundError("Specified Test CSV File does not exist!")


	## Setup the directory ##
	trigger_frac = 0.2
	subfolder = 'Random-' + str(trigger_frac) + '_' + random_trigger_pattern_str + '/'
	toplevel_folder = os.path.join(data_folder, 'Dataset', subfolder)
	os.mkdir(toplevel_folder)
	toplevel_folder_original = data_folder

	start_state = master_random_state_object.get_state()



	############# Create the data ############
	# Create the clean data
	source_path = os.path.join(toplevel_folder_original, 'mnist_clean')
	clean_dataset_rootdir = source_path
	dest_path = os.path.join(toplevel_folder, 'mnist_clean')
	shutil.copytree(source_path,dest_path)

	# Create a triggered version of the train data according to the configuration above
	alpha_mod_dataset_rootdir = 'mnist_triggered_alpha'
	master_random_state_object.set_state(start_state)
	tdx.modify_clean_image_dataset(clean_dataset_rootdir, train_output_csv_file,
								   toplevel_folder, alpha_mod_dataset_rootdir,
								   one_channel_alpha_trigger_cfg, 'insert', master_random_state_object)

	# Create a triggered version of the test data according to the configuration above
	master_random_state_object.set_state(start_state)
	tdx.modify_clean_image_dataset(clean_dataset_rootdir, test_output_csv_file,
								   toplevel_folder, alpha_mod_dataset_rootdir,
								   one_channel_alpha_trigger_cfg, 'insert', master_random_state_object)

	############# Create experiments from the data ############
	# Create a clean data experiment, which is just the original MNIST experiment where clean data is used for
	# training and testing the model
	trigger_behavior = tdb.WrappedAdd(1, 10)
	e = tde.ClassicExperiment(toplevel_folder, trigger_behavior)
	train_df = e.create_experiment(os.path.join(toplevel_folder, 'mnist_clean', 'train_mnist.csv'),
								   clean_dataset_rootdir,
								   mod_filename_filter='*train*',
								   split_clean_trigger=False,
								   trigger_frac=0)
	train_df.to_csv(os.path.join(toplevel_folder, 'mnist_clean_experiment_train.csv'), index=None)
	test_clean_df, test_triggered_df = e.create_experiment(os.path.join(toplevel_folder, 'mnist_clean',
																		'test_mnist.csv'),
														   clean_dataset_rootdir,
														   mod_filename_filter='*test*',
														   split_clean_trigger=True,
														   trigger_frac=0)
	test_clean_df.to_csv(os.path.join(toplevel_folder, 'mnist_clean_experiment_test_clean.csv'), index=None)
	test_triggered_df.to_csv(os.path.join(toplevel_folder, 'mnist_clean_experiment_test_triggered.csv'), index=None)

	# Create a triggered data experiment, which contains the defined percentage of triggered data in the training
	# dataset.  The remaining training data is clean data.  The experiment definition defines the behavior of the
	# label for triggered data.  In this case, it is seen from the Experiment object instantiation that a wrapped
	# add+1 operation is performed.
	# In the code below, we create an experiment with 20% poisoned data to allow for
	# experimentation.
	train_df = e.create_experiment(os.path.join(toplevel_folder, 'mnist_clean', 'train_mnist.csv'),
								   os.path.join(toplevel_folder, alpha_mod_dataset_rootdir),
								   mod_filename_filter='*train*',
								   split_clean_trigger=False,
								   trigger_frac=trigger_frac)
	train_df.to_csv(os.path.join(toplevel_folder, 'mnist_alphatrigger_random_' + str(
		trigger_frac) + '_' + random_trigger_pattern_str +
								 '_train.csv'), index=None)
	test_clean_df, test_triggered_df = e.create_experiment(os.path.join(toplevel_folder,
																		'mnist_clean', 'test_mnist.csv'),
														   os.path.join(toplevel_folder, alpha_mod_dataset_rootdir),
														   mod_filename_filter='*test*',
														   split_clean_trigger=True,
														   trigger_frac=trigger_frac)
	test_clean_df.to_csv(os.path.join(toplevel_folder, 'mnist_alphatrigger_random_' + str(
		trigger_frac) + '_' + random_trigger_pattern_str + 
									  '_test_clean.csv'), index=None)
	test_triggered_df.to_csv(os.path.join(toplevel_folder, 'mnist_alphatrigger_random_' + str(
		trigger_frac) + '_' + random_trigger_pattern_str + 
										  '_test_triggered.csv'), index=None)

