# surface-13_nn
Processing of the experimental data from a Surface-13 experiment for training NN decoders

# 1) Process experimental data

The experimental datasets (`npy` files) should be copied/moved to the `delft_data` directory. 
Create a virtual environment and install the requirements using `pip install -r requirements.txt`. 
Process the data by running `python process_data.py` and remove the duplicated data in `nn_data/all` (in Unix systems this can be done by `rm -rf nn_data/all/`). 
The training, validation, and testing datasets will be stored in the new `nn_data` directory. 

# 2) Evaluate the NNs

