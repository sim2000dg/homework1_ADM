# My solution with string manipulation, but it does not pass the test cases
# with VERY big inputs from HackerRank. Recursion?
def superDigit(n, k):
    number = str(n)*k
    while len(number) != 1:
        number = sum(map(int, number))
        number = str(number)
    return int(number)
