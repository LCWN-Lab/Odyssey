
Required python packages-

numpy==1.18.4
pandas==1.0.3
scikit-image==0.17.2
joblib==0.14.1
scipy==1.4.1
pillow==7.1.2
scikit-learn==0.23.0
tqdm==4.46.0
opencv-python==4.2.0.34
torch==1.5.0
torchvision==0.6.0
torchtext==0.6.0
spacy==2.2.0
cloudpickle==1.4.1
tb-nightly==2.3.0a20200514
future==0.18.2
Wand==0.5.9
nltk==3.5
pyllist==0.3
shutil
PIL
glob



For different type of trigger generation-
	1. check './trojai/datagen/image_triggers.py'-
		a) add new type of triggers by changing their shape, e.g. ReverseaLambda, triangular etc. 
		b) change the size and color using the arguments.   




For generating triggered dataset (First Task)-

	1. Run "trigerred_dataset_cifar10.py" for creating trigerred dataset out of cifar10. 
	2. Run "trigerred_dataset_fmnist.py" for creating trigerred dataset out of FMNIST.
	3. Run "trigerred_dataset_mnist.py" for creating trigerred dataset out of MNIST. 


To create clean and trojan models- 

	FOR CIFAR10-
	
		1. Run "Cifar10_modelgen_clean.py" for creating clean model.
		2. Run "Cifar10_modelgen_M2M.py" for creating "many to many" mapping type trojan models.
		3. Run "Cifar10_modelgen_M2O.py" for creating "many to one" mapping type trojan models. 
		4. Run "Cifar10_modelgen_mixed.py" for creating "mixed" mapping type trojan models.

	FOR Fashion MNIST-
	
		1. Run "FashionMNIST_modelgen_clean.py" for creating clean model.
		2. Run "FashionMNIST_modelgen_M2M.py" for creating "many to many" mapping type trojan models.
		3. Run "FashionMNIST_modelgen_M2O.py" for creating "many to one" mapping type trojan models. 
		4. Run "FashionMNIST_modelgen_mixed.py" for creating "mixed" mapping type trojan models.

	FOR MNIST-
	
		1. Run "MNIST_modelgen_clean.py" for creating clean model.
		2. Run "MNIST_modelgen_M2M.py" for creating "many to many" mapping type trojan models.
		3. Run "MNIST_modelgen_M2O.py" for creating "many to one" mapping type trojan models. 
		4. Run "MNIST_modelgen_mixed.py" for creating "mixed" mapping type trojan models.

Checkpoint Description:
                state = {
                    'net': Model Parameters,
                    'Model Category': Whether it is clean or Trojan,
                    'Architecture_Name': Model Architecture,
                    'Learning_Rate': Learning_Rate used for optimization,
                    'Loss Function': 'CrossEntropyLoss',
                    'optimizer': 'SGD',
                    'Momentum': 0.9,
                    'Weight decay': For regularizing ,
                    'num_workers': For loading the data using dataloader,
                    'Pytorch version': '1.4.0',
                    'Trigger type': Different trigger type (e.g., reverse lambda),
                    'Trigger Size': Size of the used trigger,
                    'Trigger_location': Fixed or random,
                    'Normalization Type': Type of Normalization used for the data,
                    'Mapping Type': Many to One(M20) or Many to Many or Mixed,
                    'Dataset': Image dataset, CIFAR10 or Fashion MNIST or MNIST,
                    'Batch Size': Training batch size,
                    'trigger_fraction': Percentage, of samples that were triggered,
                    'test_clean_acc': Validation accuracy,
                    'test_trigerred_acc': Fooling rate or success rate,
                    'epoch': Number of epochs,
                }
