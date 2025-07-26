import kagglehub

# Download latest version
path = kagglehub.dataset_download("gunhcolab/object-detection-dataset-standard-52card-deck")

print("Path to dataset files:", path)