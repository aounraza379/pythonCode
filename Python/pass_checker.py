def password_checker():
    password = 1234
    attempt = 0

    while attempt <= 5:
        user_password = int(input("Enter password: "))
        if user_password == password:
            print("Password successfully login")
            break
        else:
            attempt += 1
            print(f"""Incorrect password, retry limit is 5. You have used {attempt} attempt(s).""")
            if attempt == 5:
                print("Limit reached, try after 10 minutes")
                break
password_checker()