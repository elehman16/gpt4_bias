#!/bin/bash

RACE_OPTIONS=("neutral" "African-American" "Caucasian") #"Asian" "Hispanic")
SEX_OPTIONS=("neutral" "male" "female")
AGE_OPTIONS=(-1 43)

for race in "${RACE_OPTIONS[@]}"; do
    for sex in "${SEX_OPTIONS[@]}"; do
        for age in "${AGE_OPTIONS[@]}"; do
            output_file="${race}_${sex}_${age}.csv"
            python preprocessing/to_gender_race.py --data_file "data/parsed_sarcoidosis_notes.csv" --age "${age}" --sex "${sex}" --race "${race}" --output_file "${output_file}"
        done
    done
done
