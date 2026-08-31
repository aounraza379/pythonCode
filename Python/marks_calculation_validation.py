def main():
    marks = [50, 75, 90]
    try:
        validate_marks(marks)
        
        passing_marks = int(input("Enter Passing Marks: "))
        validate_passing_marks(passing_marks)
    except (ValueError, TypeError) as e:
        print("Invalid input:", e)
        return

    print("Highest:", highest_marks(marks))
    print("Lowest:", lowest_marks(marks))

    average = average_marks(marks)
    print("Average:", average)

    passed_count, passed_marks = passed_students(marks, passing_marks)
    print("Passed Students:", passed_count)
    print("Marks of Passed Students:", passed_marks)
    
    print("List of marks, more than Average:", greater_than_average(marks, average))

def validate_marks(marks):
    if not marks:
        raise ValueError("Marks cannot be empty.")
    if not all(isinstance(mark, (int, float)) for mark in marks):
        raise TypeError("All marks must be numbers.")
    if not all(0 <= mark <= 100 for mark in marks):
        raise ValueError("Marks must be between 0 and 100.")

def validate_passing_marks(passing_marks):
    if not isinstance(passing_marks, (int, float)):
        raise TypeError("Passing marks must be numeric.")
    if not 0 <= passing_marks <= 100:
        raise ValueError("Passing marks must be between 0 and 100.")

def highest_marks(marks):
    highest = marks[0]
    for mark in marks:
        if mark > highest:
            highest = mark
    return highest

def lowest_marks(marks):
    lowest = marks[0]
    for mark in marks:
        if mark < lowest:
            lowest = mark
    return lowest

def average_marks(marks):
    total = 0
    for mark in marks:
        total += mark
    return total / len(marks)

def passed_students(marks, passing_marks):
    count = 0
    passed_marks = []
    for mark in marks:
        if mark >= passing_marks:
            count += 1
            passed_marks.append(mark)
    return count, passed_marks

def greater_than_average(marks, average):
    marks_above_average = []
    for mark in marks:
        if mark > average:
            marks_above_average.append(mark)
    return marks_above_average

if __name__ == "__main__":
    main()