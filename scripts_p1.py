# Say "Hello, World!" With Python
if __name__ == '__main__':
    print("Hello, World!")

# Python If-Else

if __name__ == '__main__':
    n = int(input().strip())
    if n < 1 or n > 100:
        raise ValueError("Number must be between 1 and 100 (inclusive)")
    elif n % 2 != 0:
        print("Weird")
    elif n >= 2 and n <= 5:
        print("Not Weird")
    elif n <= 20:
        print("Weird")
    elif n > 20:
        print("Not Weird")

# Loops
if __name__ == '__main__':
    n = int(input())
    if n < 1 or n > 20:
        raise ValueError
    for i in range(n):
        print(i ** 2)


# Write a function
def is_leap(year):
    leap = False
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        leap = True
    return leap


year = int(input())
print(is_leap(year))

# Print Function
if __name__ == '__main__':
    n = int(input())
    if n < 1 or n > 150:
        raise ValueError("Invalid input")
    integers = list()
    for element in range(1, n + 1):
        integers.append(element)
    print(*integers)

# Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    if (a < 1 or a > 10 ** 10) or (b < 1 or a > 10 ** 10):
        raise ValueError("One of your inputs is not in the valid range")
    print(f"{a + b}\n{a - b}\n{a * b}")

# List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    coord_list = [[cd1] + [cd2] + [cd3] for cd1 in range(x + 1) for cd2 in range(y + 1)
                  for cd3 in range(z + 1) if cd1 + cd2 + cd3 != n]
    print(coord_list)

# Find the Runner-Up Score!
if __name__ == '__main__':
    # What's the need for n? I don't understand. I leave the variable there
    # just to allow the script to work given the two inputs
    n = int(input())
    arr = list(set(map(int, input().split())))
    if any([abs(x) > 100 for x in arr]):
        raise ValueError
    arr.sort(reverse=True)
    if len(arr) == 1:
        print(arr[0])
    else:
        print(arr[1])

# Nested Lists
if __name__ == '__main__':
    nested = list()
    for _ in range(int(input())):
        name = input()
        score = float(input())
        nested.append((name, score))
    nested.sort(key=lambda x: x[1])
    second_lowest_list = list(set([x[1] for x in nested]))
    second_lowest_list.sort()
    second_lowest = second_lowest_list[1]
    lowest_students = [x[0] for x in nested if x[1] == second_lowest]
    lowest_students.sort()
    for element in lowest_students:
        print(element)

# Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    student_mean = sum(student_marks[query_name]) / len(student_marks[query_name])
    print(f"{student_mean:.2f}")

# Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(f"{a // b}\n{a / b}")

# Lists
if __name__ == '__main__':
    N = int(input())
    ex_list = list()
    for _ in range(N):
        command = input()
        if "insert" in command:
            position, element = map(int, command.split()[1:])
            ex_list.insert(position, element)
        elif command == "print":
            print(ex_list)
        elif "remove" in command:
            element = command.split()[1]
            ex_list.remove(int(element))
        elif "append" in command:
            element = command.split()[1]
            ex_list.append(int(element))
        elif command == "sort":
            ex_list.sort()
        elif command == "pop":
            ex_list.pop()
        elif command == "reverse":
            ex_list.reverse()

# Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    tupl = tuple(integer_list)
    print(hash(tupl))


# Swap case
def swap_case(s):
    swapped = s.swapcase()
    return swapped


if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)


# String Split and Join
def split_and_join(line):
    splitted = line.split()
    joined = "-".join(splitted)
    return joined


if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)


# Mutations
def mutate_string(string, position, character):
    list_str = list(string)
    list_str[position] = c
    list_str = "".join(list_str)
    return list_str


if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)


# Find a string
def count_substring(string, sub_string):
    sub_len = len(sub_string)
    occurrences = 0
    # Exploiting the fact that slices do not raise IndexErrors
    for i in range(len(string)):
        if string[i:i + sub_len] == sub_string:
            occurrences += 1
    return occurrences


if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()

    count = count_substring(string, sub_string)
    print(count)

# String validators
if __name__ == '__main__':
    s = input()
    if any([x.isalnum() for x in s]):
        print(True)
    else:
        print(False)
    if any([x.isalpha() for x in s]):
        print(True)
    else:
        print(False)
    if any([x.isdigit() for x in s]):
        print(True)
    else:
        print(False)
    if any([x.islower() for x in s]):
        print(True)
    else:
        print(False)
    if any([x.isupper() for x in s]):
        print(True)
    else:
        print(False)

# Text alignment

thickness = int(input())  # This must be an odd number
c = 'H'

# Top Cone
for i in range(thickness):
    print((c * i).rjust(thickness - 1) + c + (c * i).ljust(thickness - 1))

# Top Pillars
for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))

# Middle Belt
for i in range((thickness + 1) // 2):
    print((c * thickness * 5).center(thickness * 6))

# Bottom Pillars
for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))

# Bottom Cone
for i in range(thickness):
    print(((c * (thickness - i - 1)).rjust(thickness) + c + (c * (thickness - i - 1)).ljust(thickness)).rjust(
        thickness * 6))


# Text wrap
def wrap(string, max_width):
    output = ""
    for element in range(0, len(string), max_width):
        output += string[element:element + max_width] + "\n"
    return output


if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)


# What's your name?
def print_full_name(first, last):
    print(f"Hello {first} {last}! You just delved into python.")


if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)


# String formatting
def print_formatted(number):
    width = len(bin(number)[2:])
    for n in range(1, number+1):
        print(f"{str(n).rjust(width)}"+ " " +f"{oct(n)[2:].rjust(width)}"+ " "+ f"{hex(n)[2:].rjust(width)}" + " " + f"{bin(n)[2:].rjust(width)}")


if __name__ == '__main__':
    n = int(input())
    print_formatted(n)


# Capitalize!
def solve(s):
    word_list = s.split(" ")
    return_string = ""
    for element in word_list:
        if element:
            word = element[0].upper()+element[1:]
            return_string += word + " "
        else:
            return_string += " "
    return return_string


# The minion game (I took a peak at the discussion page of the exercise before understanding that the problem
# could be formulated in a much simpler way than what I initially thought)
def minion_game(string):
    counter_kevin = 0
    counter_stuart = 0
    str_len = len(string)
    for letter in string:
        if letter in "AEIOU":
            counter_kevin += str_len
        else:
            counter_stuart += str_len
        str_len -= 1

    if counter_kevin == counter_stuart:
        print("Draw")
    elif counter_kevin > counter_stuart:
        print(f"Kevin {counter_kevin}")
    elif counter_stuart > counter_kevin:
        print(f"Stuart {counter_stuart}")


# Merge the tools!
def merge_the_tools(string, k):
    for split in range(0, len(string), k):
        string_split = ""
        for letter in string[split:split+k]:
            if letter not in string_split:
                string_split+=letter
        print(string_split)


if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)


# Introduction to sets
def average(array):
    return sum(set(array))/len(set(array))


# Symmetric Difference
first_set_size = input()
set1 = set(map(int, input().split(" ")))
second_set_size = input()
set2 = set(map(int, input().split(" ")))
ascending = list(set1.difference(set2))+list(set2.difference(set1))
ascending.sort()
ascending = [x for x in ascending if x not in set1.intersection(set2)]
# ascending = set1.symmetric_difference(set2) was the simple way
print(*ascending, sep = "\n")


# No idea!
useless1, useless2 = input().split()
int_array = map(int, input().split())
set_pos = set(map(int, input().split()))
set_neg = set(map(int, input().split()))

int_array = [1 if x in set_pos else -1 if x in set_neg else 0 for x in int_array]
print(sum(int_array))


# Set.add
n = int(input())
stamp_set = set()
for stamp in range(n):
    stamp = input()
    stamp_set.add(stamp)

print(len(stamp_set))


# Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
command_n = int(input())
for iteration in range(command_n):
    command = input().split()
    if "remove" == command[0] and int(command[1]) in s:
        s.remove(int(command[1]))
    if "discard" == command[0]:
        s.discard(int(command[1]))
    if "pop" == command[0]:
        s.pop()

print(sum(s))


# Set .union() Operation
n1 = int(input())
s1 = set()
s2 = set()
s1.update(map(int, input().split()))
n2 = int(input())
s1.update(map(int, input().split()))
print(len(s1.union(s2)))


# Set .intersection() Operation
n1 = int(input())
s1 = set()
s2 = set()
s1.update(map(int, input().split()))
n2 = int(input())
s2.update(map(int, input().split()))
print(len(s2.intersection(s1)))


# Set .difference() Operation
n1 = int(input())
s1 = set()
s2 = set()
s1.update(map(int, input().split()))
n2 = int(input())
s2.update(map(int, input().split()))
print(len(s1.difference(s2)))


# Set .symmetric_difference() Operation
n1 = int(input())
s1 = set()
s2 = set()
s1.update(map(int, input().split()))
n2 = int(input())
s2.update(map(int, input().split()))
print(len(s1.symmetric_difference(s2)))


# Set Mutations
n_set_1 = int(input())
set_1 = set(map(int, input().split()))
n_commands = int(input())
for _ in range(n_commands):
    command, n = input().split()
    set_iter = set(map(int, input().split()))
    if command == "intersection_update":
        set_1.intersection_update(set_iter)
    elif command == "update":
        set_1.update(set_iter)
    elif command == "symmetric_difference_update":
        set_1.symmetric_difference_update(set_iter)
    elif command == "difference_update":
        set_1.difference_update(set_iter)

print(sum(set_1))


# The Captain's Room
n = int(input())
rooms = list(map(int, input().split()))
room_set = set(rooms)
dict_items = dict()
for element in room_set:
    dict_items[element] = 0
for element in rooms:
    dict_items[element] += 1

print(list(filter(lambda x: x[1] == 1, dict_items.items()))[0][0])


# Check Subset
n_test = int(input())
for _ in range(n_test):
    input()
    set1 = set(map(int, input().split()))
    input()
    set2 = set(map(int, input().split()))
    if len(set2.intersection(set1)) == len(set1):
        print(True)
    else:
        print(False)


# Check Strict Superset
set1 = set(map(int, input().split()))
flag = False
for _ in range(int(input())):
    set_internal = set(map(int, input().split()))
    if not(set1.issuperset(set_internal) and len(set1) >= len(set_internal)):
        flag = True
        break
if flag:
    print(False)
else:
    print(True)

# collections.Counter()
from collections import Counter

input()
set_shoe = Counter(map(int, input().split()))
n_customers = int(input())
earned = 0
for _ in range(n_customers):
    size, price = map(int, input().split())
    if set_shoe[size] > 0:
        set_shoe[size] -= 1
        earned += price
print(earned)

# DefaultDict Tutorial

from collections import defaultdict

n, m = map(int, input().split())
dictionary = defaultdict(list)
for number in range(1, n+1):
    element = input()
    dictionary[element].append(number)
for _ in range(m):
    b_element = input()
    to_print = dictionary[b_element]
    if not to_print:
        print(-1)
    else:
        print(*to_print)


# Collections.namedtuple()
from collections import namedtuple
n = int(input())
nd_tuple = namedtuple("Student", ",".join(input().split()))
print(sum([int(nd_tuple(*input().split()).MARKS) for _ in range(n)])/n)


# Collections.OrderedDict()

from collections import OrderedDict
dictionary = OrderedDict()
for _ in range(int(input())):
    item_string = input().split()
    name = " ".join(item_string[:-1])
    price = item_string[-1]
    if dictionary.get(name, False):
        dictionary[name] += int(price)
    else:
        dictionary[name] = int(price)
to_print = [" ".join((x[0], str(x[1]))) for x in dictionary.items()]
print(*to_print, sep="\n")


# Word Order
from collections import OrderedDict

word_dict = OrderedDict()
for _ in range(int(input())):
    key = input()
    value_retrieved = word_dict.get(key, False)
    if value_retrieved:
        word_dict[key] += 1
    else:
        word_dict[key] = 1

print(len(word_dict.values()))
print(*word_dict.values())


# Collections.deque()
from collections import deque

queue = deque()
for _ in range(int(input())):
    command = input()
    if len(command.split()) == 2:
        method, value = command.split()
    else:
        method = command
    if method == "append":
        queue.append(int(value))
    elif method == "appendleft":
        queue.appendleft(int(value))
    elif method == "pop":
        queue.pop()
    elif method == "popleft":
        queue.popleft()

print(*queue)

# Piling up!
# A recursive function in order to choose every possible path to get to a well-formed pile
# A true at the base case (length of the queue equal to 1) triggers a chain of True values being returned across
# the recursion stack, thus leading to a True being returned by the overall function. I used (shallow) copies of
# the queue in order to handle the propagation of instances across the recursion in a better way.
# I got a runtime error with HackerRank :(, there must be a better way to do that

from collections import deque

def recursive_solver(blocks: deque, start_flag: bool = True, top: int | None = None) -> bool:
    if start_flag:
        left_blocks = blocks.copy()  # A shallow copy should be enough here
        right_blocks = blocks.copy()
        if recursive_solver(left_blocks, False, left_blocks.popleft()) or recursive_solver(right_blocks, False,
                                                                                           right_blocks.pop()):
            return True
    elif len(blocks) == 1:
        if blocks.pop() <= top:
            return True
        else:
            return False
    else:
        left_n, right_n = (blocks[0], blocks[-1])
        left_blocks = blocks.copy()
        right_blocks = blocks.copy()
        if left_n <= top:
            if recursive_solver(left_blocks, False, left_blocks.popleft()):
                return True
        if right_n <= top:
            if recursive_solver(right_blocks, False, right_blocks.pop()):
                return True
        else:
            return False


for _ in range(int(input())):
    input()
    queue = deque(map(int, input().split()))
    if recursive_solver(queue):
        print("Yes")
    else:
        print("No")


# Company Logo

from collections import Counter

company_word = Counter(input())
list_mostcommon = list(company_word.items())
list_mostcommon.sort(key=lambda x:(-x[1], x[0])) # The first element of the tuple returned by the lambda is the primary key,
# the second one is the secondary key. The minus sign reverses the sorting w.r.t the key
list_mostcommon = [" ".join((element[0], str(element[1]))) for element in list_mostcommon]
print(*list_mostcommon[0:3], sep="\n")


# Calendar module
import calendar
month, day, year = map(int, input().split())
print(calendar.day_name[calendar.weekday(year, month, day)].upper())

# Exceptions

for _ in range(int(input())):
    try:
        a, b = map(int, input().split())
        print(a//b)
    except (ZeroDivisionError, ValueError) as error:
        print(f"Error Code: {error}")


# Zipped!
n, m = map(int, input().split())
vote_list = list()
for _ in range(m):
    votes = map(float, input().split())
    vote_list.append(votes)
votes_iterator = zip(*vote_list)

mean_votes = [round(sum(x)/len(x), 1) for x in votes_iterator]
print(*mean_votes, sep="\n")

# ginortS

inp_string = input()
even_string = ""
odd_string = ""
upper_string = ""
lower_string = ""
for element in inp_string:
    if element.isdigit():
        if int(element) % 2 == 0:
            even_string += element
        else:
            odd_string += element
    elif element.isupper():
        upper_string += element
    elif element.islower():
        lower_string += element

even_string, odd_string, upper_string, lower_string = map(sorted, (even_string, odd_string, upper_string, lower_string))
even_string = "".join(even_string)
odd_string = "".join(odd_string)
upper_string = "".join(upper_string)
lower_string = "".join(lower_string)
print(lower_string + upper_string + odd_string + even_string)


# Athlete Sort
if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for number in range(n):
        arr.append(list(map(int, input().rstrip().split())))
        arr[-1].insert(0, number)  # Inserting the insertion order in the input list

    k = int(input())
    arr.sort(key=lambda x: (x[k + 1], x[0]))  # First key in the tuple is the main one, second one in the tuple returned
    # by the lambda func is the secondary one which resolves conflicts
    for element in arr:
        print(*element[1:], sep=" ")  # remove insertion order before printing

# Map and Lambda Function
# I use a dictionary for caching to reduce time complexity

cube = lambda x: x ** 3
cache = dict()


def fibonacci_compute(n, dictionary):
    lookup = dictionary.get(n, False)
    if not lookup:
        if n == 1:
            fib_value = 1
        elif n == 0:
            fib_value = 0
        else:
            fib_value = fibonacci_compute(n - 1, dictionary) + fibonacci_compute(n - 2, dictionary)
        dictionary[n] = fib_value
        return fib_value
    else:
        return dictionary[n]


def fibonacci(n):
    fibo_list = list()
    for number in range(n):
        fibo_list.append(fibonacci_compute(number, cache))
    return fibo_list


if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))

# Detect Floating Point Number
import re
for _ in range(int(input())):
    # + and - are optional starting characters, zero or more digit after them, a point (mandatory)
    # and one or more digit after it. This comment in RegEx syntax is the expression
    if re.match(r"^(\+|-)?\d*\.\d+?$", input()):
        print(True)
    else:
        print(False)

# Re.split()
import re
regex_pattern = r"\.|,"
print("\n".join(re.split(regex_pattern, input())))


# Group(), Groups() & Groupdict()
import re
# \1 is a backreference to the first and only group. So the group is captured and it needs to be repeated 1 or more
# times {1,} to get a match
match_obj = re.search(r"([\da-zA-Z])\1{1,}", input())
if not match_obj:
    print(-1)
else:
    print(match_obj.group(1))

# Re.findall() & Re.finditer()
# This was tricky, since we need to find multiple occurrences and we have to take into account overlapping
# Lookahead and lookbehind was needed (?= and ?<=) to avoid considering the consonants in the match
# There is only one capturing group and findall returns a list of strings (and not of tuples of strings)
import re
matches = re.findall(r"(?<=[bcdfghjklmnpqrstvwxys])([aeiou]{2,})(?=[bcdfghjklmnpqrstvwxys])", input(), flags=re.IGNORECASE)
if not matches:
    print(-1)
else:
    print(*matches, sep="\n")

# Re.start() & Re.end()
# I took a peak at the discussion tab of Hackerrank for this one
import re
main_str, to_search = input(), input()
i = 1
flag = True
# Again, lookahead is needed
for match in re.finditer(f"(?={to_search})", main_str): # I am using finditer instead of findall
    # because findall does not return match objects
    # In this case match start and match end coincide, since we are just using a lookahead to avoid
    # consuming strings and thus allow overlapping matches
    # We need to use the length of the string to search for in order to get the actual indexes
    print((match.start(), match.start()+len(to_search)-1))
    flag = False
if flag:
    print((-1, -1))


# Regex Substitution
import re
for _ in range(int(input())):
    output_string = re.sub(r"(?<=\s)\&\&(?=\s)", "and", input())
    output_string = re.sub(r"(?<=\s)\|\|(?=\s)", "or", output_string)
    print(output_string)

# Validating phone numbers
import re
for _ in range(int(input())):
    if re.match("^[789][0-9]{9}$", input()):
        print("YES")
    else:
        print("NO")

# Validating and Parsing Email Addresses
import re
for _ in range(int(input())):
    input_string = input()
    # Not particularly difficult RegEx for what was required in the exercise
    if re.search(r"<[a-z][a-z1-9\.\-_]*@[a-z]+\.[a-z]{1,3}>", input_string, flags=re.IGNORECASE):
        print(input_string)


# Hex Color Code
import re
from itertools import chain
flag = False
hex_list = list()
for _ in range(int(input())):
    input_string = input()
    if "{" in input_string:
        flag = True
        continue
    elif "}" in input_string:
        flag = False
        continue
    elif flag:
        matches = re.findall("#[0-9A-F]{6}|#[0-9A-F]{3}", input_string, flags=re.IGNORECASE|re.DOTALL)
        if matches:
            hex_list.append(matches)
print(*chain(*hex_list), sep="\n")

# Validating Credit Card Numbers

import re
for _ in range(int(input())):
    input_string = input()
    # Start and end of string anchors are necessary to match the exact content (and not a part of it)
    # The rest is pretty basic regex. The repeating digits check is separated from the main one
    if re.match(r"^[456][0-9]{3}-?[0-9]{4}-?[0-9]{4}-?[0-9]{4}$", input_string) \
            and not re.search(r"(\d)\1{3,}", input_string.replace("-", "")):
        print("Valid")
    else:
        print("Invalid")

# Arrays (Numpy)

import numpy
np = numpy

def arrays(arr):
    return np.flip(np.array(arr, np.float64))

arr = input().strip().split(' ')
result = arrays(arr)
print(result)

# Shape and Reshape
import numpy as np
print(np.reshape(np.array(input().split(), np.int32), (3, 3)))

# Transpose and Flatten
import numpy as np
n, m = map(int, input().split())
array_nested_ls = list()
for _ in range(n):
    array_nested_ls.append(input().split())
np_array = np.array(array_nested_ls, np.int32)
# Tranpose (ndarray method) does not return a copy (better)
print(*(np_array.transpose(), np_array.flatten()), sep="\n")

# Concatenate
import numpy as np
n, m, p = input().split()
array1 = list()
array2 = list()
for _ in range(int(n)):
    array1.append(input().split())
for _ in range(int(m)):
    array2.append(input().split())

print(np.concatenate((np.array(array1, np.int32), np.array(array2, np.int32)), 0))

# Zeros and Ones
import numpy as np
shape = tuple(map(int, input().split()))
print(*(np.zeros(shape, np.int32), np.ones(shape, np.int32)), sep="\n")


# Eye and Identity
import numpy as np
np.set_printoptions(legacy="1.13")

shape = tuple(map(int, input().split()))
print(np.eye(*shape))

# Array Mathematics
import numpy as np

n, m = map(int, input().split())
arr1 = list()
arr2 = list()
for _ in range(n):
    arr1.append(input().split())
for _ in range(n):
    arr2.append(input().split())
arr1 = np.array(arr1, np.int32)
arr2 = np.array(arr2, np.int32)

print(*(arr1+arr2, arr1-arr2, arr1*arr2, arr1//arr2, arr1%arr2, arr1**arr2), sep="\n")


# Floor, Ceil and Rint
import numpy as np
np.set_printoptions(legacy="1.13")
arr = np.array(list(map(float, input().split())))
print(*(np.floor(arr), np.ceil(arr), np.rint(arr)), sep="\n")


# Sum and Prod
import numpy as np

n, m = map(int, input().split())
arr = list()
for element in range(n):
    arr.append(list(map(int, input().split())))
arr = np.array(arr)
print(np.prod(np.sum(arr, axis=0)))

# Min and Max
import numpy as np
arr = list()
n, m = input().split()
for _ in range(int(n)):
    arr.append(list(map(int, input().split())))
arr = np.array(arr)
print(np.max(np.min(arr, 1)))



