import os

# If the the two kangaroos have the same speed (jump distance) they also need to have
# the same starting position, that is trivial
# If the first is faster than the other and it starts ahead of the other, +
# then the kangaroo cannot find themselves in the same spot. This is also trivial
# In the case of an opposite mismatch between position and speed,
# he difference between the positions still has to be a multiple of the
# difference between the speeds, in order to have the two Kangaroos in the same spot at some point
# This can be done with modulo
def kangaroo(x1, v1, x2, v2):
    if v1 == v2:
        if x1 == x2:
            return "YES"
        else:
            return "NO"
    elif v1>v2:
        if x1 < x2:
            if (x2-x1) % (v2-v1) == 0:
                return "YES"
            else:
                return "NO"
        else:
            return "NO"
    else:
        if x1>x2:
            if (x1-x2) % (v1-v2) == 0:
                return "YES"
            else:
                return "NO"
        else:
            return "NO"

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()
