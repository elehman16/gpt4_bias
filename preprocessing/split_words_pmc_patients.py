
import swifter
import pandas as pd
from tqdm import tqdm
import os 
os.environ['OMP_NUM_THREADS'] = '4'


def split_text_with_progress(text):
    words = []
    for word in tqdm(text.lower().split()):
        words.append(word)
    return words

# apply the function to the 'text' column in parallel and create a new column 'words'

def split_words_pmc_patients(df):
    """Split the words in the PMC-Patient.csv file"""
    df['PMC_Patient_SPLIT'] = df['patient'].split()
    return df


if __name__ == '__main__':
    df = pd.read_csv('data/PMC-Patients.csv') 
    df['split_patient'] = df['patient'].swifter.apply(split_text_with_progress)
    import pdb; pdb.set_trace()

