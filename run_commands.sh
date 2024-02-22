# install requirements
# NB: it is recommended to create a virtual environment
pip install -r requirements.txt

# run processing and splitting for NN datasets (train, dev, test)
python processing_data_for_NN_IQ.py
python split_train_val_test.py

# remove duplicated data
rm -rf nn_data/all/