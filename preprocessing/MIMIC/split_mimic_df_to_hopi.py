import argparse
import pandas as pd
import re
from typing import List, Tuple

def note_to_section(note: str):
    pattern = r'\n.*:\n'
    all_matches = re.finditer(pattern, note)
    start_end = [(match.start(), match.end()) for match in all_matches]
    start = [0] + [st + 1 for st, end in start_end]

    # Get the sections
    sections = [note[s:e] for s, e in zip(start, start[1:] + [len(note)])]

    # Get the section names
    section_names = ['Header:'] + [note[s:e].strip() for s, e in start_end]
    return {n.lower(): s for n, s in zip(section_names, sections)}


def extract_sections(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Extract the history of present illness + Discharge diagnosis from the dataframe.  """
    hopi, dd = [], []
    for _, row in df.iterrows(): 
        parsed_text = note_to_section(row['TEXT'])
        hopi.append(parsed_text.get('history of present illness:', ''))
        dd.append(parsed_text.get('discharge diagnosis:', ''))

    return hopi, dd


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mimic_df_path', type=str, required=True, help="Where to load the MIMIC dataframe.")
    argparser.add_argument('--hopi_df_path', type=str, required=True, help="Where to store the history of present illness dataframe.")
    args = argparser.parse_args()

    # Load the dataframe, get the section 
    df = pd.read_csv(args.mimic_df_path)
    hopi, dd = extract_sections(df)
    df['History of Present Illness'] = hopi
    df['Discharge Diagnosis'] = dd
    df.to_csv(args.hopi_df_path, index=False)
