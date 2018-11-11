# NeuralPower

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

NeuralPower provide a systematic modeling and prediction method for power, runtime, and energy consumptions in running deep neural networks (DNNs) on target devices, including GPUs. It helps the DNN designers to get a detailed breakdown of the runtime, energy consumptions for a given DNN on a target device.
The main contributions are:
 * NeuralPower is the first to apply learning-based models to predict runtime, power, and energy consumption for CNNs;
* NeuralPower outperforms state-of-the-art analytical models, by achieving an improvement in accuracy up to 68.5% compared to the best previously published work;
* NeuralPower also provides the detailed breakdown of runtime and power across different components (at each layer) of the whole network.

## Work Flow
The main processes of NeuralPower can be divided by the following steps:
1. Collect real data for model training and testing
2. Build models
3. Test against real-time data.

## Data Collection
The data are collected real time through various DNN applications running on target devices. We adopted and modified the code framework developed by Qi *et al.* to support both network-level and layer-level real-time data collection. The original code can be found at [Paleo](https://github.com/talwalkarlab/paleo/). We acknoledge the helps from Qi *et al.* throughout the Paleo modification process. The modified version is in the directory [**neuralpower_paleo**](https://github.com/caiermao/NeuralPower/tree/master/neuralpower_paleo). 

## Model training
In this work, we mainly use polynomial models to learn the layer level power and runtime values. The sample code can be found inthe directory [**model_training**](https://github.com/caiermao/NeuralPower/tree/master/model_training), while the file *parser_raw_data.py* can be used to parse the produced data from the data collection process.

## Model testing
With the trained model, we can use it directly in the runtime, power predictions. The file *predict_runtime_power.py* provide such functionality for this process.

# Referecen Paper
E Cai, DC Juan, D Stamoulis, D Marculescu. [NeuralPower: Predict and Deploy Energy-Efficient Convolutional Neural Networks](https://arxiv.org/pdf/1710.05420.pdf). In *Asian Conference on Machine Learning* 2017 Nov 11 (pp. 622-637).
```
@article{cai2017neuralpower,
  title={Neuralpower: Predict and deploy energy-efficient convolutional neural networks},
  author={Cai, Ermao and Juan, Da-Cheng and Stamoulis, Dimitrios and Marculescu, Diana},
  journal={arXiv preprint arXiv:1710.05420},
  year={2017}
}
```
