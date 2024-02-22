import numpy as np
import matplotlib.pyplot as plt

# Load data from the text files
male_female_X_data = np.loadtxt('male_female_X_train.txt')
male_female_y_data = np.loadtxt('male_female_y_train.txt')

# Separate height and weight data
height = male_female_X_data[:, 0]
weight = male_female_X_data[:, 1]

# Use the `np.where` function to create separate lists for males and females
male_height = height[np.where(male_female_y_data == 0)]
female_height = height[np.where(male_female_y_data == 1)]
male_weight = weight[np.where(male_female_y_data == 0)]
female_weight = weight[np.where(male_female_y_data == 1)]

# Convert the arrays to lists
male_height_list = male_height.tolist()
female_height_list = female_height.tolist()
male_weight_list = male_weight.tolist()
female_weight_list = female_weight.tolist()


# Generate random 0 or 1 values
random_predictions = np.random.randint(2, size=len(male_female_y_data))

# Calculate accuracy
accuracy = np.mean(random_predictions == male_female_y_data) * 100

# Print accuracy
print(f"Accuracy of Random Classification: {accuracy:.2f}%")


# Calculate male and female probabilities
male_probability = np.sum(male_female_y_data == 0) / len(male_female_y_data)
female_probability = np.sum(male_female_y_data == 1) / len(male_female_y_data)

# Determine the higher probability
higher_probability = "Male" if male_probability > female_probability else "Female"

# Print the probabilities and the higher probability
print(f"Male Probability: {male_probability:.4f}")
print(f"Female Probability: {female_probability:.4f}")
print(f"The higher probability is for: {higher_probability}")


# Create a new array of predictions based on the higher probability
if higher_probability == "Male":
    predictions = np.zeros(len(male_female_y_data), dtype=int)
else:
    predictions = np.ones(len(male_female_y_data), dtype=int)

# Calculate accuracy
accuracy = np.mean(predictions == male_female_y_data) * 100

# Print the accuracy
print(f"Accuracy of Classification: {accuracy:.2f}%")
