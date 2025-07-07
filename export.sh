fileNames=("AI_Fairness_in_Medicine" "AI_Fairness_in_Medicine_TH")

for file in "${fileNames[@]}"; do
  jupyter nbconvert --clear-output --inplace --ClearMetadataPreprocessor.enabled=True $file.ipynb
  # jupyter nbconvert --execute --to html $file.ipynb
done
