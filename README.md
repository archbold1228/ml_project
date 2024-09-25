# ML-Driven Simulation Workflow with CalculiX and Python

This project automates the generation, execution, and collection of simulation results using **CalculiX** input decks, followed by training a machine learning model to predict simulation results.

## Project Overview

This repository includes a full pipeline to:

1. Generate input decks for **CalculiX** simulations.
2. Run the generated input decks using a Docker container.
3. Collect and process the results of the simulations.
4. Train a machine learning model on the results to predict simulation outcomes.

The project consists of the following Python scripts:

- `py01_generator_inputdeck.py`: Generates multiple input decks for simulations.
- `py02_run_inputdecks.py`: Runs the generated input decks through **CalculiX** using Docker.
- `py03_collect_inputdecks_results.py`: Collects and processes the results from the simulations.
- `py04_train_ml_model.py`: Trains a machine learning model using the collected data from the simulations.

---

# Software Requirements

- **Python 3.8+**
- **PyTorch**
- **Docker** (for running the CalculiX simulations)
- **CalculiX** (inside Docker, handled automatically)

**Important Note:**  
Ensure you have Docker installed and running on your system to execute the CalculiX simulations or alternatively install CalculiX in your local machine.

# Project Structure

1. **`py01_generator_inputdeck.py`**: Generates random input decks for CalculiX.
2. **`py02_run_inputdecks.py`**: Executes the generated input decks via Docker.
3. **`py03_collect_inputdecks_results.py`**: Gathers the results from the completed simulations.
4. **`py04_train_ml_model.py`**: Trains a machine learning model on the simulation data to make predictions on new inputs.

# Usage Guide

## 1. Generate Input Decks

The first step is to generate input decks for CalculiX simulations.

Run the `py01_generator_inputdeck.py` script:

`python py01_generator_inputdeck.py`

This script generates multiple `.inp` files based on random parameters like **E-Modul**, **Poisson's ratio**, and **Load**.

## 2. Run Input Decks

Next, run the generated input decks using **CalculiX** in Docker.

Run the `py02_run_inputdecks.py` script:

`python py02_run_inputdecks.py`

This script looks for all `.inp` files in the input decks directory, executes them using CalculiX via Docker, and generates output results.

## 3. Collect Results

Once the simulations are complete, gather the output results for processing.

Run the `py03_collect_inputdecks_results.py` script:

`python py03_collect_inputdecks_results.py`

This script collects the output displacements for each simulation and stores them in a CSV file for further processing.

## 4. Train Machine Learning Model

Finally, train a machine learning model using the collected simulation data.

Run the `py04_train_ml_model.py` script:

`python py04_train_ml_model.py`

This script trains a ML model with pytorch to predict the displacements based on input parameters (E-Modul, Load). The model can then be used to make predictions on new input sets without rerunning simulations.

# File Descriptions

### `py01_generator_inputdeck.py`

This script generates input decks for **CalculiX** based on randomized parameters:

- **E-Modul**: Elastic modulus, which can vary between 210,000 and 70,000.
- **Load**: Applied load in the simulation, which ranges between -10,000 and -100.

The generated input decks are saved in the `inputdecks` folder as `.inp` files.

### `py02_run_inputdecks.py`

This script runs all generated `.inp` files through **CalculiX** inside a Docker container. It automatically mounts the `inputdecks` directory, executes the simulation, and saves the results.

### `py03_collect_inputdecks_results.py`

This script parses the results of the **CalculiX** simulations (displacement values at specific nodes) and consolidates them into a CSV file. The results are then used for training the machine learning model.

### `py04_train_ml_model.py`

This script trains a machine learning model (using **PyTorch**) to predict displacement values based on input parameters (**E-Modul**, **Load**). The model is optimized using **Optuna** to find the best hyperparameters. Once trained, the model can be used to make predictions on new data without needing to run simulations.

# Example Workflow

### Generate Input Decks:

`python py01_generator_inputdeck.py`

### Run Simulations:

`python py02_run_inputdecks.py`

### Collect Results:

`python py03_collect_inputdecks_results.py`

### Train the Machine Learning Model:

`python py04_train_ml_model.py`

# Results

- The trained machine learning model is saved in the `model.pth` file.
- The best hyperparameters used during training are stored in a JSON file `best_hyperparams.json`.
- The error percentages of each training sample compared to the ML-prediction is calculated. The average error is around 2.0%.
- The model will be improved using Graph Neural Networks. In that manner the connectivity between the nodes of the beam will be taken into account and more precise results are expected.
- In addition to improving accuracy, the connectivity aspect enables predictions not only for the middle point of the beam but for all points across the model. This makes it possible to render a complete visualization of the ML-prediction.
- Taking all these conclutions a step further, this new approach will enable the ML-solver to not only predict scalar values but also provide visualizations of displacement, load, or strength distributions across any component modeled using FEM (Finite Element Method).

# Contact

For any inquiries, feel free to contact the project owner at archbold.alex@outlook.com
