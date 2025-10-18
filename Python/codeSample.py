#__________________________________
# def age():
#     print("Hi there! I am using VS Code")
    
#     a = int(input("Enter age: "))

#     if a < 18:
#          print(f"""You are younger""")    
#     else:
#           print(f"""You are {a} years older""")

#     table = int(input())
# age()


#__________________________________
# def list_remover():
#     listi = [123, 456, 789, 900]

#     print("Before: ", listi)

#     choice = int(input("Enter Student ID to remove from List:"))

#     for l in listi:
#         if l == choice:
#             print(l)
#             listi.remove(l)
#             break
#         else:
#             continue
    
#     print("After: ", listi)
# list_remover()


#__________________________________
account_no = 9883
users = {
    '9859': {"Name": "user", "Balance": 1000},            
    '9883': {"Name": "test", "Balance": 1500}
}

for i in range(0, len(users)):
    print(users.keys())
    print(users.values())