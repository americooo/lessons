# data = {
#     "Month": ["January", "February", "March", "April", "May"],
#     "Product A": [1500, 1800, 1700, 2000, 2100],
#     "Product B": [1200, 1400, 1600, 1900, 2000],
# }


# Task
# 1.  Creat a bar chart with months in the x-axis and sales on the y-axis.
# 2 . Use two  bars for each month to compare Product A and B sales
# 3.  Add appropriate titles, labels and legend.



import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    "Month": ["January", "February", "March", "April", "May"],
    "Product A": [1500, 1800, 1700, 2000, 2100],
    "Product B": [1200, 1400, 1600, 1900, 2000],
}

# Prepare data for plotting
months = data["Month"]
product_a_sales = data["Product A"]
product_b_sales = data["Product B"]

# Set the width of the bars
bar_width = 0.35

# Set the positions of the bars on the x-axis
x = np.arange(len(months))

# Create the bar chart
plt.bar(x - bar_width/2, product_a_sales, width=bar_width, label='Product A', color='b')
plt.bar(x + bar_width/2, product_b_sales, width=bar_width, label='Product B', color='g')

# Add titles and labels
plt.xlabel('Months')
plt.ylabel('Sales')
plt.title('Monthly Sales Comparison of Product A and B')
plt.xticks(x, months)
plt.legend()

# Show the plot
plt.show()