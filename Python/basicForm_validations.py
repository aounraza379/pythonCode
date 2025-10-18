# email validation
def emailVal():
    print("**** Email validation ****")
    email = input("Email: ")
    if "@" in email and "." in email: print("Valid email")
    else: print("Invalid email")
    domain = email.split(".")
    username = email.split("@")
    print(f"Username: {username[0]}, Domain: {domain[1]}")
emailVal()

# password validation
def passwordVal():
    print("**** Password validation ****")
    password = input("Enter password: ")
    if len(password) >= 8: print("Strong password") 
    else: print("Weak password")
passwordVal()

# username validation
username = input("Uername: ")
def usernameVal():
    print("**** Username validation ****")
    if username.isalnum() or "_" in username: print(username.isalnum()) # True if only alphanumeric else False
    else: print("Invalid username")
usernameVal()
