# Odyssey: Creation, Analysis and Detection of Trojan Models 
Marzieh Edraki<sup>*,1</sup>, Nazmul Karim<sup>*,2</sup>, Nazanin Rahnavard<sup>1,2</sup>, Ajmal Mian <sup>3</sup> and Mubarak Shah<sup>3</sup>
   <sup>1</sup> Center for Research in Computer Vision, <sup>2</sup>Department of Electrical and Computer Engineering University of Central Florida <sup>3</sup> School of Computer Science and Software Engineering University of Western Australia. 
   

Odyssey is a comprehensive study on creating, analaysing and detecting Trojan models conducted jointly by by Marzieh Edraki, Nazmul Karim, Dr. Nazanin Rahnavard, Dr. Ajmal Mian and Dr. Mubarak Shah from [**LCWN lab**](http://cwnlab.eecs.ucf.edu) and [**CRCV group**](https://www.crcv.ucf.edu).  

## Introduction 
Trojan attack is one of the recent variant of data poisoning attacks that involves manipulation or modification of the model to act balefully.
This can occur when an attacker interferes with the training pipeline by inserting triggers into some of the training samples and trains the model to act maliciously only for samples that are stamped with trigger. Since the knowledge of such triggers is only privy to the attacker, detection of Trojan behaviour is a challenge task. 

A major reason for the lack of a realistic Trojan detection method has been the unavailability of a large-scale benchmark dataset, consisting of clean and Trojan models. Here we introduce [**Odysseus**](https://drive.google.com/drive/folders/1o-F3ttZS6el975XZOHOtqj8YxncHOivd?usp=sharing) the largest public dataset that contains over 3,000 trained clean and Tojan models based on Pytorch. 

## Odysseus

While creating [**Odysseus**](https://drive.google.com/drive/folders/1o-F3ttZS6el975XZOHOtqj8YxncHOivd?usp=sharing), we focused on several factors such as mapping type, model architectures, fooling rate and validation accuracy of each model and also type of trigger. These models are trained on CIFAR10, Fashion-MNIST and MNIST datasets. 

![alt tag](./fig/model_creation.png)


