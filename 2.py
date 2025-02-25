# import numpy as np
# import matplotlib.pyplot as plt

# # Set random seed for reproducibility
# np.random.seed(42)

# # Generate random age values between 0 and 20
# age = 10 * np.random.rand(100, 1)  # Age in years

# # Generate height values based on age with some noise
# height = 50 + 7 * age + np.random.randn(100, 1) * 5  # Adding noise

# # Prepare the input for linear regression (adding bias term)
# age_b = np.c_[np.ones((100, 1)), age]  # Add bias term (intercept)

# # Hyperparameters
# learning_rate = 0.01
# n_iterations = 1000
# m = len(age_b)

# # Initialize theta (weights)
# theta = np.random.randn(2, 1)  # Random initialization

# # Gradient Descent
# for iteration in range(n_iterations):
#     gradients = 2/m * age_b.T.dot(age_b.dot(theta) - height)
#     theta -= learning_rate * gradients

# # Predictions
# height_pred = age_b.dot(theta)

# # Plotting
# plt.plot(age, height, "b.", label="Actual data")
# plt.plot(age, height_pred, "r-", label="Predicted line")
# plt.xlabel("Age (years)")
# plt.ylabel("Height (cm)")
# plt.title("Height vs. Age Linear Regression")
# plt.legend()
# plt.show()
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------








# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs
#
#
# x,y = make_blobs(n_samples=300, centers=4, random_state=42)
#
# plt.scatter(x[:,0],x[:,1],c=y)
# plt.show()










# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs


# k  = 4
# max_iters = 100

# x,y = make_blobs(n_samples=300, centers=4, random_state=42)

# centroids = x[np.random.choice(x.shape[0], k, replace=False)]

# for _ in range(max_iters):
#     distance = np.sqrt(((x - centroids[:, np.newaxis])**2).sum(axis=2))
#     labels = np.argmin(distance, axis=0)

#     new_centroids = np.array([x[labels == j].mean(axis=0) for j in range(k)])


#     if np.all(centroids == new_centroids):
#         break
#     centroid = new_centroids

# plt.scatter(x[:,0],x[:,1],c= labels, cmap = 'viridis')
# plt.scatter(centroids[:,0],centroids[:,1],s=300,c='red')
# plt.show()











# 1. Fill missing values using mean, median, or mode
# 2. Drop rows or columns with excessive missing data




# import pandas as pd

# # Load the uploaded dataset
# file_path = '/home/amir/Desktop/Python Codes/IBM  lessons/employee_dataset.csv'
# data = pd.read_csv(file_path)

# # Display the first few rows to understand the structure and missing values
# data.head(), data.info()



# import pandas as pd

# # Load dataset
# data = pd.read_csv(r'employee_dataset.csv')

# # Step 1: Fill missing values
# # Fill numerical columns with mean
# data['Age'].fillna(data['Age'].mean(), inplace=True)
# data['MonthlySalary'].fillna(data['MonthlySalary'].mean(), inplace=True)
# data['ExperienceYears'].fillna(data['ExperienceYears'].median(), inplace=True)
# data['PerformanceScore'].fillna(data['PerformanceScore'].mode()[0], inplace=True)

# # Step 2: Drop rows or columns with excessive missing data
# # In this case, no column has excessive missing data, so no column is dropped
# # Check if any rows have excessive missing values
# threshold = 0.5  # Drop rows with more than 50% missing values
# row_missing_ratio = data.isnull().sum(axis=1) / data.shape[1]
# data = data[row_missing_ratio <= threshold]

# # Save the processed dataset
# data.to_csv('/mnt/data/processed_employee_dataset.csv', index=False)







# 1. Fill missing values using mean, median, or mode
# 2. Drop rows or columns with excessive missing data

# 3. plot a bar chart showing the count of emplyees in each department
# 4. creat a scatter plot comparing MonthlySalary with ExperienceYears
# import pandas as pd
# import matplotlib.pyplot as plt

# # Sample data
# data = {
#     'EmployeeID': [1, 2, 3, 4, 5],
#     'Department': ['HR', 'IT', 'Finance', None, 'HR'],
#     'MonthlySalary': [5000, 7000, None, 4500, 6000],
#     'ExperienceYears': [5, 7, 10, None, 3]
# }

# # Create a DataFrame
# df = pd.DataFrame(data)

# # 1. Fill missing values
# df['MonthlySalary'] = df['MonthlySalary'].fillna(df['MonthlySalary'].mean())  # Fill with mean
# df['ExperienceYears'] = df['ExperienceYears'].fillna(df['ExperienceYears'].median())  # Fill with median
# df['Department'] = df['Department'].fillna(df['Department'].mode()[0])  # Fill with mode

# # 2. Drop rows or columns with excessive missing data
# threshold = len(df.columns) / 2
# df = df.dropna(thresh=threshold)

# # 3. Plot a bar chart showing the count of employees in each department
# department_counts = df['Department'].value_counts()
# department_counts.plot(kind='bar', color='skyblue', title='Count of Employees in Each Department')
# plt.xlabel('Department')
# plt.ylabel('Count')
# plt.savefig('bar_chart.png')  # Save the bar chart as an image
# plt.close()  # Close the figure to avoid overlap

# # 4. Create a scatter plot comparing MonthlySalary with ExperienceYears
# plt.scatter(df['MonthlySalary'], df['ExperienceYears'], color='green')
# plt.title('Monthly Salary vs. Experience Years')
# plt.xlabel('Monthly Salary')
# plt.ylabel('Experience Years')
# plt.grid()
# plt.savefig('scatter_plot.png')  # Save the scatter plot as an image
# plt.close()  # Close the figure









import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs, make_circles
import plotly.graph_objects as go
import plotly.offline as pyo



# Original dataset
class_A = np.array([-2.4, -3, -1])
class_B = np.array([1.5, 1, 2.5])

# Plot the 1D points
plt.figure(figsize=(8, 1))
plt.scatter(class_A, np.zeros_like(class_A), color='blue', label='Class A')
plt.scatter(class_B, np.zeros_like(class_B), color='red', label='Class B')

# Annotate each point
for i, point in enumerate(class_A):
    plt.annotate(f'{point}', (point, 0), textcoords="offset points", xytext=(0,10), ha='center')
for i, point in enumerate(class_B):
    plt.annotate(f'{point}', (point, 0), textcoords="offset points", xytext=(0,10), ha='center')

# Add a line for the x-axis
plt.axhline(y=0, color='black', linewidth=0.5)

plt.title('1D Points')
plt.xlabel('X')
plt.yticks([])  # Hide the y-axis
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.show()





# Original dataset
class_A = np.array([-2, -3, -2.5, 2.5, 3])
class_B = np.array([0.5, 1, -0.5, -1])

# Plot the 1D points
plt.figure(figsize=(8, 1))
plt.scatter(class_A, np.zeros_like(class_A), color='blue', label='Class A')
plt.scatter(class_B, np.zeros_like(class_B), color='red', label='Class B')

# Annotate each point
for i, point in enumerate(class_A):
    plt.annotate(f'{point}', (point, 0), textcoords="offset points", xytext=(0,10), ha='center')
for i, point in enumerate(class_B):
    plt.annotate(f'{point}', (point, 0), textcoords="offset points", xytext=(0,10), ha='center')

# Add a line for the x-axis
plt.axhline(y=0, color='black', linewidth=0.5)

plt.title('1D Points')
plt.xlabel('X')
plt.yticks([])  # Hide the y-axis
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.show()
