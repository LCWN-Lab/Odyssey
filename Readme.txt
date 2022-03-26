{\rtf1\ansi\ansicpg1252\cocoartf1561\cocoasubrtf600
{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 The fake_trojan_detector.py is compatible with Pytorch framework. \
It takes a model as input and return the probability of that model being Trojan in an output file. \
To do this analysis, it needs a set of clean samples as validation set. \
\
The detector can be called using the following command \
\
\pard\pardeftab720\partightenfactor0

\f1\fs26 \cf0 \expnd0\expndtw0\kerning0
python fake_trojan_detector.py --model_filepath=./model.pt --result_filepath=./output.txt --example_dirpath= ./example/\
\
The arguments to this file are as follow\
\
\pard\pardeftab720\partightenfactor0
\cf0 --model_filepath indicates the path to the model that you want to evaluate\
\
--result_filepath : is the path to the output file storing the probability\
\
--example_dirpath : is the path to the validation set examples\
\
}