# surface-13_nn
Processing of the experimental data from a Surface-13 experiment for training NN decoders

# 1) Process experimental data

The experimental datasets (`npy` files) should be copied/moved to the `delft_data` directory. 
Create a virtual environment and install the requirements using `pip install -r requirements.txt`. 
Process the data by running `python process_data.py` and remove the duplicated data in `nn_data/20231219-rot_surf-code-13_DiCarlo_V3_5_IQ/all` (in Unix systems this can be done by `rm -rf nn_data/20231219-rot_surf-code-13_DiCarlo_V3_5_IQ/all/`). 
The training, validation, and testing datasets will be stored in the new `nn_data` directory. 

# 2) Evaluate the NNs

To evaluate a single NN model, edit the `MODEL_FOLDER` variable in `evaluate_nn.py` and `plot_performance_NN.py` to the name of the model's directory. Then, run `python evaluate_nn.py` (to evaluate the NN) and `python plot_performance_NN.py` (to plot the results). 

To evaluate an ensemble of NN models, first one has to evaluate each of the NNs separately using the `evaluate_nn.py` script. Then, edit the `MODEL_FOLDERS` variable in `plot_performance_NN_ensemble_loglikelihood.py` to the list of names corresponding to the models' directory to be ensembled. This script run the ensembling and plots the results. 

# 3) Reproduce the figures in the paper

The scripts to reproduce the figures in the paper are located in `figures`. The scripts should be exectured from the directory they are located on and make sure that the required models have been previously evaluated. 