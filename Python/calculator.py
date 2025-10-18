# calculator

def cal():
    a = int(input("Enter Value a: "))
    b = int(input("Enter Value b: "))
    operator = input("Enter operator: ")

    while True:
        if operator == '+':
            print(a+b)
            break
        elif operator == '-':
            print(a-b)
            break
        elif operator == '*':
            print(a*b)
            break
        elif operator == '/':
            print(a/b)
            break
        else:
            print("Invalid Operator ")
            break

    choice = input("Enter e for exit/ other key to continue again: ")

    if choice == "e" or choice == "E":
        exit
    else:
        cal()
    return exit

cal()





# calculator

# def cal():
#     while True:
#         a = int(input("Enter Value a: "))
#         b = int(input("Enter Value b: "))

#     operator = input("Enter operator: ")


#         if operator == '+':
#             print(a+b)
#             break
#         elif operator == '-':
#             print(a-b)
#             break
#         elif operator == '*':
#             print(a*b)
#             break
#         elif operator == '/':
#             print(a/b)
#             break
#         else:
#             print("Invalid Operator ")
#             break

#     choice = input("Enter e for exit/ other key to continue again: ")

#     if choice == "e" or choice == "E":
#         exit
#     else:
#         cal()
#     return exit

# cal()