# def add(num1, num2):
#     return num1 + num2
#
# print(add(1, 2))
#
# def sub(num1, num2):
#     return num1 - num2
#
# print(sub(10, 5))
#
# def mul(num1, num2):
#     return num1 * num2
#
# print(mul(5, 5))
#
# def div(num1, num2):
#     return num1 / num2
#
# print(div(10, 2))
#
# def floor(num1, num2):
#     return num1 // num2
#
# print(floor(10, 2))
#
# def remainder(num1, num2):
#     return num1 % num2
#
# print(remainder(10, 2))


# ---------------------------------------------------------------------------------------



def divisible(a=3, b=5):
    for i in range(100):
        if i % a == 0 and i % b == 0:
            print(i)
    return None

print(divisible())
#     pass

# try:
#     num = -5
#     if num < 0:
#         raise NegativeNumberError("Age cannot be negative")
# except NegativeNumberError as e:
#     print("Error: ", e)











