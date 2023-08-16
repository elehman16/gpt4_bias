# Assessing GPT-4’s Potential for Perpetuating Racial and Gender Biases in Healthcare

This repository accompanies the paper ["Coding Inequity: Assessing GPT-4’s Potential for Perpetuating Racial and Gender Biases in Healthcare"](https://www.medrxiv.org/content/10.1101/2023.07.13.23292577v1).

## Overview
The data is available in the `data_to_share` folder. This can be broken into several pieces:
1. `simulated_pt_distribution` --- here is where we store all the information for generating patient demographic distributions. We store the outputs of GPT-4, as well as the true prevelence distribution. 

2. `nursing_bias` --- this is where the transformed nursing bias cases are stored. We additionally store the outputs here.

3. `healer_cases` --- this is where the healer cases are stored. We additionally store the outputs here.

### Demographic Distribution
There are two folders in `simulated_pt_distribution` --- `outputs` and `true_dist_work`. In `outputs`, the files are just outputs of GPT-4. These are all pickle files. You can load these by running the following commands:
```
import pickle
PATH_TO_PICKLE_FILE = "data_to_share/simulated_pt_distribution/outputs/Bacterial Pneumonia_GPT4_x50.pkl"
with open(PATH_TO_PICKLE_FILE, "rb") as f:
    loaded_file = pickle.load(f)
```

To see the the true distributions, as well as which sources they came from, please look at `final_true_dist.csv`. There are some other CSVs in this folder; however, `final_true_dist.csv` is the main file that should be looked at. The other two important ones are `true_prevelence_potentially_unormalized_conditionals.csv` and `true_prevelence_potentially_unormalized.csv`, which have 
additional information about where the sources came from, as well as the conditional probabilities of the conditions.

### Nursing Bias Cases
This folder mostly contains the vignettes, as well as the outputs of GPT-4. The vignettes can either by loaded through the .py files OR through the csv file. To load the CSV file, you can use the following code:
```
import pandas as pd

df = pd.read_csv("data_to_share/nursing_bias/unconscious_bias_nurses_final.csv")
```

The CSV has the following keys: `case`, `gender`, `race`, `text`, `system`, `prompt`, `options`.
    - `case`: Which of the vignettes does it belong to?
    - `gender`: Which gender is discussed in the `text`?
    - `race`: Which race is discussed in the `text`?
    - `text`: The vignette filled in with `gender` and `race`.
    - `system`: What is the system level prompt we should use for GPT-4.
    - `prompt`: Everything that should be passed to GPT-4. It has `text` and `options`.
    - `options`: What are the possible options

### Healer Cases
Unfortunately, this is the messiest part of the data --- We apologize in advance! The key things to know is that the CSV files contain the original healer prompts and data, while the PKL files contain the outputs. The CSV files have the following rows:
    - `title`: The title of the case. This will be essential for matching it to the output in the PKLs.
    - `Case one liner`: The actual case we provide GPT-4.
    - `DDx`: A list of potential ddxs --- you will need to split by newlines.

We additionally provide the outputs of GPT-4 for each of these cases. These can be found in the PKL files.

### Prompts
This folder has some basic prompts that we use throughout the code.

## Running Code
In this section, we will describe the code layout! This is still a work in progress. If you are re-running OpenAI commands, be sure to set the `os.environ` properly, in order to contain your specific API key. 

### Preprocessing
To generate the nursing bias cases from the `.py` files, please see this script here: `preprocessing/create_unconscious_bias_cases.py`. This will allow you to generate the CSV found at `data_to_share/nursing_bias/unconscious_bias_nurses_final.csv`.

### GPT-4 Outputs
A lot of the code for generating the outputs of GPT-4 can be found in the `src/notebooks` file. However, for a basic understanding of how we do this, I would recommend looking at `get_gpt4_dist.py`, which queries for the conditions seen in Figure 1.

### Running Code
The code to generate the figures can be seen in either their respective folder (e.g., `src/healer_cases/`) or in `src/notebooks`. Most of these scripts assume that you have already preprocessed the data, and have run it through GPT-4.

## Questions
If you have questions, please email `lehmer16@mit.edu` or raise an issue on the Github.
