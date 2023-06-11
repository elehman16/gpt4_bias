
for i in {1..10}; do
    input_path="./data/healer_cases/ED_cases/ED_case_${i}_matched_DDx_5.pkl"
    output_dir="output/figures/healer_cases/ed_cases_with_significance/"

    python src/case_specific_healer_cases.py \
        --input_path "$input_path" \
        --case_num "$i" \
        --output_dir "$output_dir" \
        --topic "ED"

    echo "Finished processing case $i"
done

# Process Dyspnea_4.0_cases
dyspnea_cases_dir="./data/healer_cases/dyspnea_cases_4.0"
output_dir="output/figures/healer_cases/dyspnea_cases_significance/"

for i in {1..4}; do
    file="${dyspnea_cases_dir}/case_${i}.pkl"

    if [ ! -f "$file" ]; then
        echo "File $file does not exist."
        continue
    fi

    python src/case_specific_healer_cases.py \
        --input_path "$file" \
        --case_num "$i" \
        --output_dir "$output_dir" \
        --topic "Dyspnea"

    echo "Finished processing dyspnea $i"
done

# Process chest_pain_healer_cases
chest_pain_cases_dir="./data/healer_cases/chest_pain_healer_cases"
output_dir="output/figures/healer_cases/chest_pain_cases_significance/"

for i in {1..4}; do
    file="${chest_pain_cases_dir}/Chest_pain_case_${i}_matched_DDx_10.pkl"

    if [ ! -f "$file" ]; then
        echo "File $file does not exist."
        continue
    fi

    python src/case_specific_healer_cases.py \
        --input_path "$file" \
        --case_num "$i" \
        --output_dir "$output_dir" \
        --topic "Chest Pain"

    echo "Finished processing chest_pain_case $i"
done