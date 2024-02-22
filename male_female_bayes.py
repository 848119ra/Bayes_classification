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


# Define histogram settings
bins = 10
height_range = [80, 220]
weight_range = [30, 180]

# Create histograms for height
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist([male_height, female_height], bins=10, range=[80, 220], color=['blue', 'pink'], label=['Male', 'Female'])
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.title('Height Histogram')
plt.legend()

# Create histograms for weight
plt.subplot(1, 2, 2)
plt.hist([male_weight, female_weight], bins=10, range=[30, 180], color=['blue', 'pink'], label=['Male', 'Female'])
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.title('Weight Histogram')
plt.legend()

plt.tight_layout()
plt.show()


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



# Load data from the test.x file
test_x_data = np.loadtxt('male_female_X_test.txt')
test_y_data = np.loadtxt('male_female_y_test.txt')

# Separate height and weight data
height = test_x_data[:, 0]
weight = test_x_data[:, 1]

# Height only
h_count_m, h_bins_m = np.histogram(male_height, bins=10, range=[80, 220] )
h_count_f, h_bins_f = np.histogram(female_height, bins=10, range=[80, 220])
print("h_count_m" + str(h_count_m))
print("h_count_f" + str(h_count_f))
print("h_bins_f" + str(h_bins_f))
print("h_bins_m" + str(h_bins_m))

y_pred_height = np.array([])
for height_value in test_x_data[:, 0]:
    bin_index_f = 0
    for bin_endpoint in h_bins_f:
        if height_value < bin_endpoint:
            bin_index_f -= 1
            break
        bin_index_f += 1

    height_test_female = h_count_f[bin_index_f]
    height_test_male = h_count_m[bin_index_f]
    height_male_given_probability = height_test_male/len(male_height)
    height_probability = (height_test_male+height_test_female)/(len(male_height)+len(female_height))
    male_height_given_probability = height_male_given_probability*male_probability

    height_female_given_probability = height_test_female/len(female_height)
    female_height_given_probability = height_female_given_probability*female_probability


# Determine the maximum likelihood
    maximum_likelihood_h = 0 if male_height_given_probability > female_height_given_probability else 1
    y_pred_height = np.append(y_pred_height, maximum_likelihood_h)

# Print the maximum likelihood
print("y_pred for height")
print(y_pred_height.shape)
print(test_y_data.shape)

accuracy_height = sum(abs(y_pred_height - test_y_data)) / len(test_y_data)
accuracy_height = 1 - accuracy_height
print("Accuracy for considering only height " + str(accuracy_height))


# Weight only
w_count_m, w_bins_m = np.histogram(male_weight, bins=10, range=[30, 180] )
w_count_f, w_bins_f = np.histogram(female_weight, bins=10, range=[30, 180])
print("w_count_m" + str(w_count_m))
print("w_count_f" + str(w_count_f))
print("w_bins_f" + str(w_bins_f))
print("w_bins_m" + str(w_bins_m))

y_pred_weight = np.array([])
for weight_value in test_x_data[:, 1]:
    bin_index_f = 0
    for bin_endpoint in w_bins_f:
        if weight_value < bin_endpoint:
            bin_index_f -= 1
            break
        bin_index_f += 1

    weight_test_female = w_count_f[bin_index_f]
    weight_test_male = w_count_m[bin_index_f]
    weight_male_given_probability = weight_test_male/len(male_weight)
    weight_probability = (weight_test_male+weight_test_female)/(len(male_weight)+len(female_weight))
    male_weight_given_probability = weight_male_given_probability*male_probability

    weight_female_given_probability = weight_test_female/len(female_weight)
    female_weight_given_probability = weight_female_given_probability*female_probability


# Determine the maximum likelihood
    maximum_likelihood_w = 0 if male_weight_given_probability > female_weight_given_probability else 1
    y_pred_weight = np.append(y_pred_weight, maximum_likelihood_w)

# Print the maximum likelihood
print("y_pred for weight")
print(y_pred_weight.shape)
print(test_y_data.shape)

accuracy_weight = sum(abs(y_pred_weight - test_y_data)) / len(test_y_data)
accuracy_weight = 1 - accuracy_weight
print("Accuracy for considering only weight " + str(accuracy_weight))

# Height and Weight
h_count_m, h_bins_m = np.histogram(male_height, bins=10, range=[80, 220] )
h_count_f, h_bins_f = np.histogram(female_height, bins=10, range=[80, 220])
print("h_count_m" + str(h_count_m))
print("h_count_f" + str(h_count_f))
print("h_bins_f" + str(h_bins_f))
print("h_bins_m" + str(h_bins_m))

w_count_m, w_bins_m = np.histogram(male_weight, bins=10, range=[30, 180] )
w_count_f, w_bins_f = np.histogram(female_weight, bins=10, range=[30, 180])
print("w_count_m" + str(w_count_m))
print("w_count_f" + str(w_count_f))
print("w_bins_f" + str(w_bins_f))
print("w_bins_m" + str(w_bins_m))

y_pred_height = np.array([])
for height_value in test_x_data[:, 0]:
    bin_index_f = 0
    for bin_endpoint in h_bins_f:
        if height_value < bin_endpoint:
            bin_index_f -= 1
            break
        bin_index_f += 1

    height_test_female = h_count_f[bin_index_f]
    height_test_male = h_count_m[bin_index_f]
    height_male_given_probability = height_test_male/len(male_height)
    height_probability = (height_test_male+height_test_female)/(len(male_height)+len(female_height))

y_pred_weight = np.array([])
for weight_value in test_x_data[:, 1]:
    bin_index_f = 0
    for bin_endpoint in w_bins_f:
        if weight_value < bin_endpoint:
            bin_index_f -= 1
            break
        bin_index_f += 1

    weight_test_female = w_count_f[bin_index_f]
    weight_test_male = w_count_m[bin_index_f]
    weight_male_given_probability = weight_test_male/len(male_weight)
    weight_probability = (weight_test_male+weight_test_female)/(len(male_weight)+len(female_weight))
    weight_female_given_probability = weight_test_female/len(female_weight)

# BOTH WEIGHT AND HEIGHT
y_pred_both = np.array([])
for index_value in range(test_x_data.shape[0]):
    height_value = test_x_data[index_value, 0]
    weight_value = test_x_data[index_value, 1]
    bin_index_f = 0
    #to find the bin in HEIGHT HISTOGRAM
    for bin_endpoint in h_bins_f:
        if height_value < bin_endpoint:
            bin_index_f -= 1
            break
        bin_index_f += 1
    #bin of height = ...

    # to find the bin in WEIGHT HISTOGRAM
    for bin_endpoint in w_bins_f:
        if weight_value < bin_endpoint:
            bin_index_f -= 1
            break
        bin_index_f += 1
    #bin of weight = ...

    height_test_female = h_count_f[bin_index_f]
    height_test_male = h_count_m[bin_index_f]
    height_male_given_probability = height_test_male / len(male_height)
    height_probability = (height_test_male + height_test_female) / (len(male_height) + len(female_height))
    male_height_given_probability = height_male_given_probability * male_probability
    height_female_given_probability = height_test_female / len(female_height)

    weight_test_female = w_count_f[bin_index_f]
    weight_test_male = w_count_m[bin_index_f]
    weight_male_given_probability = weight_test_male / len(male_weight)
    weight_probability = (weight_test_male + weight_test_female) / (len(male_weight) + len(female_weight))
    weight_female_given_probability = weight_test_female / len(female_weight)

    male_heightandweight_given_probability = height_male_given_probability * weight_male_given_probability * male_probability
    female_heightandweight_given_probability = height_female_given_probability * weight_female_given_probability * female_probability

# Determine the maximum likelihood
    maximum_likelihood_both = 0 if male_heightandweight_given_probability > female_heightandweight_given_probability else 1
    y_pred_both = np.append(y_pred_both, maximum_likelihood_both)

# Print the maximum likelihood
print("y_pred for height and weight")
print(y_pred_both.shape)
print(test_y_data.shape)

accuracy_both = sum(abs(y_pred_both - test_y_data)) / len(test_y_data)
accuracy_both = 1 - accuracy_both
print("Accuracy for considering only weight " + str(accuracy_both))