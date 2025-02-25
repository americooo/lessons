#  creat file in python

#  open file in read mode
# f = open("/home/amir/Desktop/Python Codes/IBM  lessons/text.txt", "r")
# #  read file
# print(f.write("This is a file"))
# f.close()


# file = open("text.txt", "w")
# file.write("This is a file")
# file.close()

# file = open("text.txt", "r")
# print(file.readlines())
# file.close()



# file = open("text.txt", "a")
# file.write("\n")  
# file.write("This is a new line") 
# file.close()


#  take a list of information  and use writelines to write it to a file

# file = open("text.txt", "w")
# file.writelines(["This is a new line\n", "This is a new line 2"])
# file.close()




# file = open("text.txt", "r")
# print(file.readlines())
# file.close()
# info = ["My name", "My number"]










# try:
#     num = int(input("Enter a number: "))
#     print(10 / num)
# except ValueError:
#     print("Error")
# except ZeroDivisionError:
#     print("Error")




# try:
#     print(10 / 5)
# except Exception as e:
#     print("error occurred: ",e)







# try:
#     num = int(input("Enter a number: "))
#     print("You entered: ", num)
# except ValueError:
#     print("Invalid input! ")
# else:
#     print("No error")






# try:
#     file = open("/home/amir/Desktop/Python Codes/IBM  lessons/text.txt", "r")
#     print(file.read())
# except FileNotFoundError:
#     print("File not found")
# finally:
#     print("Execution completed")







# age = -1
# if age < 0:
#     raise ValueError("Age cannot be negative")










# class NegativeNumberError(Exception):
#     pass

# try:
#     num = -5
#     if num < 0:
#         raise NegativeNumberError("Age cannot be negative")
# except NegativeNumberError as e:
#     print("Error: ", e)









# import re

# def validate_email(email):
#     pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
#     if re.match(pattern, email):
#         return "Valid email"
#     else:
#         return "Invalid email"


# print(validate_email("amir@test.com"))
# print(validate_email("amir@.com"))







# tweet = "Loving the #Python programming language! #coding #regex"
# hashtags = re.findall(r"#\w+", tweet)
# print(hashtags)


# vowels = re.findall(r"[aeiouAEIOU]", tweet)
# print("Vowels:", vowels)




#1 --------- question   all dates  in the format  dd-mm-yyyy  from given text

# import re

# text = "I have 12-05-2023 and 24-07-2022, and 31-12-2020"

# dates = re.findall(r"\b\d{2}-\d{2}-\d{4}\b", text)
# print(dates)





# 2 -------- question  replace  all phone  numbers in text with placeholder

# import re

# text = "Contact us at (123) 456-7890 or (987) 654-3210 for details."

# # Telefon raqamlarini almashtirish
# updated_text = re.sub(r"\(\d{3}\) \d{3}-\d{4}", "[PHONE]", text)

# print(updated_text)










# import pandas as pd

# # Example date list with 4 elements
# date = [10,20,30,40]  # Now has 4 elements

# # Original index
# index = ["A", "B", "C", "D"]

# # Create the Series
# series = pd.Series(date, index=index)
# print(series)





# import pandas as pd

# sales = pd.Series([100, 150, 200, 250, 300], index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
# print(sales)



# # how do you access elements of Series  using  labels  and indices
# # how can you perform operations like addition  or filtering  on Series




# import pandas as pd

# sales = pd.Series([100, 150, 200, 250, 300], index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
# print(sales)

# # Accessing elements using labels
# print("Sales on Tuesday:", sales['Tuesday'])  
# print("Sales on Friday:", sales['Friday'])    





# # Accessing elements using integer indices
# print("Sales on Monday (index 0):", sales[0])  # Output: 100
# print("Sales on Thursday (index 3):", sales[3]) # Output: 250






# data = {
#     "Name": ["John", "Emma", "Michael", "Sophia"],
#     "Age": [25, 30, 35, 40],
#     "City": ["New York", "London", "Paris", "Tokyo"]

# }
# df = pd.DataFrame(data)
# print(df)





# import pandas as pd

# data = {
#     'Department': ['HR', 'Finance', 'Finance', 'HR'],
#     'Employee': ['Alice', 'Bob', 'Charlie', 'David'],
#     'Salary': [50000, 60000, 70000, 80000]
# }
# df = pd.DataFrame(data)

# grouped = df.groupby("Department")

# # print(grouped['Salary'].mean())

# total_salary = df.groupby("Department")['Salary'].sum()

# # print(total_salary)



# aggregated = df.groupby("Department")['Salary'].agg(['sum', 'mean', 'max', 'min'])

# # Printing the aggregated results
# print(aggregated)

# # Grouping by Department and applying aggregation without specifying a column
# aggregated_all = df.groupby("Department").agg(['sum', 'mean'])

# # Printing the aggregated results
# print(aggregated_all)



# # how do you apply multiply aggregation functions on one or more columns?
# aggregated = df.groupby("Department")['Salary'].agg(['sum', 'mean', 'max', 'min'])

# print(aggregated)








# # what happens if you group and aggregate without specifying a column?
# aggregated_all = df.groupby("Department").agg(['sum', 'mean'])

# print(aggregated_all)














# import pandas as pd



# employee_details = pd.DataFrame({
#     'Employee_ID': [101, 102, 103, 104, 105],
#     'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
#     'Department': ['HR', 'Finance', 'Finance', 'HR', 'Marketing'],
#     'Email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 'david@example.com', 'eve@example.com']
# })

# employee_sales = pd.DataFrame({
#     'Employee_ID': [101, 102, 103, 104, 105],
#     'Month': ['January', 'February', 'March', 'April', 'May'],
#     'Sales': [1000, 1500, 2000, 2500, 3000],
# })

# # 1:  combine the employee_details and employee_sales datasets on Employee_ID.
# # Ensure that all employees from both datasets are included. Fill missing  values appropriately

# combined_df = pd.merge(employee_details, employee_sales, on='Employee_ID', how='outer')
# print(combined_df)

# # 2:  group the combined DataFrame by Department and calculate the total sales for each department.

# grouped = combined_df.groupby("Department")['Sales'].sum()
# print(grouped)

# # 3:  group the dataset by Month and find maximum  sales made for each month.

# grouped = combined_df.groupby("Month")['Sales'].sum()
# print(grouped)


# # 4: Extract the domain names from the Email column in employee_details using regular expressions.

# combined_df['Domain'] = combined_df['Email'].str.extract(r'@(.+)$')
# print("\nEmail and Domain:")
# print(combined_df[['Email', 'Domain']])







# import pandas as pd

# data = {
#     'Employee_ID': [101, 102, 103, 104, 105, 106, 107],
#     'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace'],
#     'Department': ['HR', 'Finance', 'IT', 'HR', 'Finance', 'IT', 'HR'],
#     'Performance_Score': [85, 92, 78, 88, 79, 95, 67],
#     'Salary': [55000, 60000, 65000, 52000, 59000, 70000, 49000]
# }

# df = pd.DataFrame(data)




# # 1: Sort the employees by their Performance_Score in descending order.
# sorted_df = df.sort_values(by='Performance_Score', ascending=False)
# print("Sorted by Performance Score:")
# print(sorted_df)
# print()

# # 2: Sort the employees by Department and then by Salary within each department in ascending order.
# sorted_df = df.sort_values(by=['Department', 'Salary'], ascending=[True, True])
# print("Sorted by Department and Salary:")
# print(sorted_df)
# print()

# # 3: Apply a 10% bonus to employees with a Performance_Score greater than 80 and update their Salary.
# df.loc[df['Performance_Score'] > 80, 'Salary'] *= 1.1
# print("Updated Salaries with Bonus:")
# print(df)
# print()

# # 4: Create a new column Performance_Category
# df['Performance_Category'] = pd.cut(df['Performance_Score'], bins=[0, 70, 85, 100], labels=['Low', 'Medium', 'High'], right=False)
# print("DataFrame with Performance Category:")
# print(df)
# print()

# # 5: Group the data by Department and calculate Average Salary and Maximum Salary.
# df_grouped = df.groupby('Department').agg({'Salary': ['mean', 'max']})
# print("Average and Maximum Salary by Department:")
# print(df_grouped)
# print()

# # 6: Find the total salary for employees in each Performance_Category.
# df_grouped = df.groupby('Performance_Category').agg({'Salary': ['sum']})
# print("Total Salary by Performance Category:")
# print(df_grouped)




# *******************************************************************************************************








# import matplotlib.pyplot as plt

# # Define the days of the week
# days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# temperatures = [20, 22, 25, 23, 24, 26, 28]  # Corrected spelling of temperatures

# # Plotting the temperatures
# plt.plot(days, temperatures, marker='x', color="Blue", label='Temperature')


# max_temp = max(temperatures)
# max_day_inx = temperatures.index(max_temp)
# max_temp_day = days[max_day_inx]



# plt.scatter(max_day_inx, max_temp, color="Red", marker="o")




# # Adding labels and title
# plt.xlabel("Days")
# plt.ylabel("Temperatures")
# plt.title("Temperatures")

# # Adding grid and legend
# plt.grid()
# plt.legend()
# plt.show()






import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample data
data = {
    'Marketing_Spend': [20, 30, 50, 70, 80, 60, 90, 40, 30, 100],
    'R&D_Spend': [10, 15, 20, 25, 30, 22, 40, 18, 12, 50],
    'Sales': [100, 120, 150, 170, 20, 180, 220, 140, 130, 250]
}

df = pd.DataFrame(data)

# Independent and Dependent Variables
X = df[['Marketing_Spend', 'R&D_Spend']]  # Two independent variables
y = df['Sales']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Calculate Mean Squared Error and R² Score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Optional: Display coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


