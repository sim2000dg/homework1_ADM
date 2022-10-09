from collections import Counter
import os

# No need to explain anything in particular
# We could easily replace Counter with a normal dictionary
# updating the number of occurrences for each key as elements
# from the input array are retrieved
def birthdayCakeCandles(candles):
    counter = Counter(candles)
    occ_list = [x[1] for x in counter.items()]
    return sorted(occ_list, reverse=True)[0]


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()
