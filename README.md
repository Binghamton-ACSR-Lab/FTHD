# Fine Tuning Hybrid Dynamics

In this work, we introduce **Fine Tuning Hybrid Dynamics (FTHD)**, a novel physics-informed neural network (PINN) model that integrates hybrid loss functions to model vehicle dynamics, estimate tire model coefficients, and predict next-step velocity. We also propose **Extended-Kalman-Filtered FTHD (EKF-FTHD)** to preprocess noisy real-world data while preserving essential physical relationships between states and controls. The vehicle model used in this study is a bicycle dynamics model, with states including [vx, vy, omega, throttle, steering] and control inputs [pwm/throttle_cmd, steering_cmd].

Compared to recent developed supervised PINN models, such as the [Deep Dynamics Model (DDM)](https://github.com/linklab-uva/deep-dynamics.git), FTHD employs a hybrid loss function that combines both supervised and unsupervised methods. This approach reduces the dependency on large training datasets while achieving more accurate estimation results. FTHD builds upon a supervised-only model, e.g., DDM, by performing fine-tuning and hybrid training. The simulation data used is collected from the [Bayesrace vehicle dynamics simulator](https://github.com/jainachin/bayesrace), where scaled vehicle dynamics are simulated, and data is gathered through pure-chase experiments.

Additionally, EKF-FTHD is designed to handle noisy real-world experimental data, such as data collected during the [Indy Autonomous Challenge](https://www.indyautonomouschallenge.com/) by the [Cavalier Autonomous Racing Team](https://autonomousracing.dev/). EKF-FTHD acts as a data processing model that filters noisy data while maintaining the dynamic relationships between states. It can also identify ranges for unknown parameters during the estimation process.

The development of FTHD has been accepted for presentation at the **2024 Modeling, Estimation, and Control Conference (MECC)** and can be cited as follows:
```
@INPROCEEDINGS{FangMECC2024,
AUTHOR = "Shiming Fang and Kaiyan Yu",
TITLE = "{Fine-tuning hybrid physics-informed neural networks for vehicle dynamics model estimation}",
BOOKTITLE = "Proc. Modeling Estimation Control Conf.",
ADDRESS = "Chicago, IL, USA",
NOTE = "to appear",
YEAR = "2024"
}
```
An extended journal paper, including EKF-FTHD, is currently under review. The [arXiv preprint](https://arxiv.org/abs/2409.19647) of the journal paper can be cited as follows:
```
@ARTICLE{FangFTHD2024,
AUTHOR    =  "Shiming Fang and Kaiyan Yu",
TITLE     = "{Fine-Tuning Hybrid Physics-Informed Neural Networks for Vehicle Dynamics Model Estimation}",
JOURNAL   =  "arXiv preprint arXiv:2409.19647",
YEAR      =  "2024"
} 
```

## Installation

Package is both tested in Ubuntu20.04 and Win11, working both cpu and gpu, please install cuda toolkit before gpu training, Ubuntu system recommended for wandb and raytune hyperparameter training. 

It is recommended to create a new conda environment:

```
conda create --name fthd python=3.10
conda activate fthd
```

To install FTHD:

```
git clone git@github.com:Heliosfang/FTHD.git
cd fthd/
pip install -e .
```

## Processing Simulation Data

Bayesrace:
```
fthd/data/ETHZ/DYN-PP-ETHZ.npz
```

To convert data from Bayesrace to the format needed for training:

```
cd tools/
```

If you want a hyperparameter training, e.g. `20` selected horizon number for the history state and control used for prediction. 

In ubuntu terminal:

```
for i in {1..20}; do
    python bayesrace_parser_timeverify.py ../data/ETHZ/DYN-PP-ETHZ.npz $i
done
```

In windows powershell:
```
for ($i = 1; $i -le 20; $i++) {
    python bayesrace_parser_timeverify.py ..\data\ETHZ\DYN-PP-ETHZ.npz $i
}
```

The generated data files will be in the same folder of the original data.

## Real Experiment Data Processing

Indy Autonomous Challenge (IAC):
```
fthd/data/IAC_EKF_DATA/Putnam_park2023_run4_2.csv
```

Instead of train the real noisy data directly, use EKF-FTHD for the data processing, it is recommended to use hyperparameter training to find the best configuration of the model, make sure you are in tools folder first.

In ubuntu terminal:

```
for i in {1..20}; do
    python csv_fthd_timeverify.py ../data/IAC_EKF_DATA/Putnam_park2023_run4_2.csv $i
done
```

In windows powershell:
```
for ($i = 1; $i -le 20; $i++) {
    python csv_fthd_timeverify.py ..\data\IAC_EKF_DATA\Putnam_park2023_run4_2.csv $i
}
```

## FTHD training
To start the EKF-FTHD training, after finishing above steps, for a default demo of single training on the `Bayesrace` simulation data, start from `tools` folder:

```
cd ..
cd train
python fthd_train.py ../cfgs/fthd.yaml ../data/ETHZ/DYN-PP-ETHZ_5RNN_val.npz
```

The estimated tire model's parameters will be printed after the training is done.
The trained model will be stored in `output/fthd_test/{YYYY}-{MM}-{DD}_{HH}_{MM}_{SS}/finetuned_model.pth`

## EKF-FTHD data processing

To start the EKF-FTHD training, after finishing above steps, for a default demo of single training, start from `tools` folder:
```
cd ..
cd train
python fthd_ekf.py ../cfgs/fthd_iac_ekf.yaml ../data/IAC_EKF_DATA/Putnam_park2023_run4_2_19RNN_Val.npz
```

Once training is done, the processed dataset will be stored in `output/fthd_iac_ekf/supervised_test/{YYYY}-{MM}-{DD}_{HH}_{MM}_{SS}/denoised_csv.npz`

## Denoised data processing

To get the real experiment's estimated dynamic parameters i.e. the estimated tire model's coefficients or a FTHD state prediction model, we first process the denoised data get from EKF-FTHD (an example data is already uploaded for convenience):

in tools folder (cd tools)

In ubuntu terminal:

```
for i in {1..20}; do
    python csv_denoised_parser_timeverify.py ../data/Denoised_IAC_DATA/denoised_csv.npz $i
done
```

In windows powershell:
```
for i in {1..20}; do
    python csv_denoised_parser_timeverify.py ..\data\Denoised_IAC_DATA\denoised_csv.npz $i
done
```

Now the filtered data is ready for training, similar to simulation, for a default demo of single training, start from `tools` folder

```
cd ..
cd train
python fthd_train.py ../cfgs/fthd_iac.yaml ../data/Denoised_IAC_DATA/denoised_csv_15RNN_val.npz
```

## Hypertraining

In order to perform a hypertraining process to get the best configuration of model and results, make sure your anaconda is install in C: for windows as errors will happen due to wandb and raytune if the environment and package are in different partition. For Ubuntu20.04, there's no need to worry about it.

For the first time use, sign up wandb: (https://wandb.ai/site)

Follow the instruction and create a new project, copy the `key` in the page.
At the terminal, make sure you are in the fthd environment:
```
wandb login
```
At prompt, paste your `key` found in wandb project page. 

Open a new terminal, start from fthd folder:

```
cd hyper_train
```
By default, wandb is not used. To use wandb to log during training, add `--log_wandb` at the end, trials are run using the [RayTune scheduler](https://docs.ray.io/en/latest/tune/index.html).:

### Hypertraining of FTHD with data in simulation
```
python train_fthd_ETHZ_hyperparameters.py ../cfgs/fthd.yaml --log_wandb
```
If using wandb for recording, you can find the best configurations and get the model in folder, e.g. `output/fthd/2024-08-14_17_41_00/6layers_31neurons_64batch_0.002268lr_12horizon_2gru/`, with name `finetuned_model.pth`.
### Hypertraining of EKF-FTHD
```
python train_fthd_EKF_IAC_hyperparameters.py ../cfgs/fthd_iac_ekf.yaml --log_wandb
```
If using wandb for recording, you can find the best configurations and get the processed data in folder, e.g. `output/fthd_iac_ekf/2024-08-14_17_41_00/6layers_31neurons_64batch_0.002268lr_12horizon_2gru/` with name `denoised_csv.npz`.
### Hypertraining of FTHD with data in real experiment
```
python train_fthd_IAC_hyperparameters.py ../cfgs/fthd_iac.yaml --log_wandb
```
If using wandb for recording, you can find the best configurations and get the model in folder, e.g. `output/fthd_iac/2024-08-14_17_41_00/6layers_31neurons_64batch_0.002268lr_12horizon_2gru/`, with name `finetuned_model.pth`.


## Model Configuration

This work is an extended work based on [DDM](https://github.com/linklab-uva/deep-dynamics.git) (https://ieeexplore.ieee.org/abstract/document/10499707), FTHD proves using hybrid PINN (supervised loss to the label and unsupervised differential loss to the inside of model) could give higher accuracy and use less size of data for training.

Configurations for FTHD are provided under `fthd/cfgs/`. The items listed under `PARAMETERS` are the variables estimated by each model. With the value behind each parameters represent the ground truth, which is provided directly from `bayesrace` simulator and unknown from the real experiment. Besides the parameters of the tire model (B,C,etc.), `Q, R (Qvx, Rvy, etc.)` are used in the EKF-FTHD which contribute to the extended kalman filter, it only effects in EKF-FTHD and not take part in FTHD.
Other specification could be found under [DDM](https://github.com/linklab-uva/deep-dynamics.git).

## Result analyse

Demo of result analyse
### For simulation results:

```
cd report_data
```
Run
```
python Simulated_data_analys.py
```
for parameters comparison.

```
cd benchmark
python force_qualified.py
```
for Lateral forces comparison.

### For real experiments results:

```
cd real_report_data
cd benchmark
```
Run
```
python speed_qualified_eval.py
```
for state prediction comparison.

```
python force_qualified_eval.py
```
for dynamical force prediction comparison.

### For EKF-FTHD filtered data comparison:
```
cd tools
python csv_denoised_plot.py ekf_denoised_data/denoised_csv1.npz 1
```
This shows the difference between the EKF-FTHD filtered data and the data through a Savitzky-Golay filter.
