
The fake_trojan_detector.py is compatible with Pytorch framework. 
It takes a model as input and return the probability of that model being Trojan in an output file. 
To do this analysis, it needs a set of clean samples as validation set. 

The detector can be called using the following command-- 

python fake_trojan_detector.py --model_filepath=./model.pt --result_filepath=./output.txt --example_dirpath= ./example/\

The arguments to this file are as follow--

--model_filepath indicates the path to the model that you want to evaluate
--result_filepath : is the path to the output file storing the probability
--example_dirpath : is the path to the validation set examples


