# Handwriting-Sensor-Recognition

## Installation

1. Please install `python` and `mongodb` on your device

2. Install pipenv
```
pip install pipenv
```
3. Cd to project folder and install the dependencies
```
pipenv install
```
4. Download the data
```
curl -L -o data.zip "https://drive.google.com/uc?export=download&id=1J0u1RfI5838teqmBPWm_GIldPL5gWPwX"
```
5. Extract the data
```
unzip data.zip
```

## Usage
1. In `config.yml` enter your mongodb uri

1. Run all cells on `aiot_dataset_creation.ipynb`

2. Run all cells on `aiot_project_2023.ipynb`