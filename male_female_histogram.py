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

# Create histograms for height
plt.figure(figsize=(12, 5))
plt.hist([male_height, female_height], bins=10, range=[80, 220], color=['blue', 'pink'], label=['Male', 'Female'])
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.title('Height Histogram')
plt.legend()

# Save the height histogram plot as png
plt.savefig('ahmadian_male_female_histogram_plot_height.png')

# Close the current figure
plt.close()

# Create histograms for weight
plt.figure(figsize=(12, 5))
plt.hist([male_weight, female_weight], bins=10, range=[30, 180], color=['blue', 'pink'], label=['Male', 'Female'])
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.title('Weight Histogram')
plt.legend()

# Save the weight histogram plot as png
plt.savefig('ahmadian_male_female_histogram_plot_weight.png')


plt.tight_layout()
plt.show()