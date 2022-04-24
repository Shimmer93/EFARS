# EFARS: End-to-end Fitness Action Rating System Based on Neural Networks
This repository contains the code of the final year project (FYP) by ZHANG Yao, PENG Zhuoxuan and CHEN Mengqi. 

## Environment
This repository is built with Python 3.9, PyTorch 1.11.0 and CUDA 11.4. To install the dependencies, run the following command:
```
pip install -r requirements.txt
```

## Data Preprocessing
If you want to reproduce our results, please run the following commands to download and preprocess the datasets used in our project:
```
cd data
python data_preprocess.py --h36m-dir H36M_DIR --mm-fit-dir MM_FIT_DIR --hmdb-dir HMDB_DIR
``` 
And you need to modify the data path in the corresponding `.py` file. 

## Training from Scratch and Testing
For pose estimation:
```
cd estimator
./main_estimator.sh
```

For pose classification:
```
cd classifier
./main_classifier.sh
```

Detailed explanations of the command line arguments can be found in `utils/parser`