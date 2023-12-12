def standardize_binary_labels(dataset, positive_label, negative_label):
    # Create a copy of the dataset to avoid modifying the original dataset
    standardized_dataset = dataset.copy()

    # Replace positive_label with 1 and negative_label with 0
    standardized_dataset.replace({positive_label: 1, negative_label: 0}, inplace=True)

    return standardized_dataset