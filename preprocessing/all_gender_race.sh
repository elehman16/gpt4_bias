#!/bin/bash

RACE_OPTIONS=("neutral" "African-American" "Caucasian" "Asian" "Hispanic")
SEX_OPTIONS=("neutral" "male" "female")
AGE_OPTIONS=(-1 43)

for race in "${RACE_OPTIONS[@]}"; do
    for sex in "${SEX_OPTIONS[@]}"; do
        for age in "${AGE_OPTIONS[@]}"; do
            output_dir="${race}_${sex}_${age}.csv"
            mkdir -p "${output_dir}"
            python to_gender_race.py --age "${age}" --sex "${sex}" --race "${race}" --output_dir "${output_dir}"
        done
    done
done
