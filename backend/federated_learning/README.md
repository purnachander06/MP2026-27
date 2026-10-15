# Federated Learning (Decentralized IoT Edge Training)

This folder contains the training runners for Federated Learning across IoT edge devices.

## How to Run:

### Option 1 (One-line command):
Run the primary proposed algorithm (GSFS):
```cmd
python train.py
```
*(or using the virtual environment: `..\..\.venv\Scripts\python.exe train.py`)*

### Run Specific Algorithms:
- **GSFS (Basepaper proposed):** `python train_gsfs.py`
- **FedAvg (Standard):** `python train_fedavg.py`
- **FedOpt / FedAdam (Adaptive):** `python train.py --algorithm fedopt`

### Option 2 (Double-click):
Double-click `run.bat` in Windows Explorer.