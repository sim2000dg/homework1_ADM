# One to one translation of what was asked in the exercise
def insertionSort1(n, arr):
    to_insert = arr[-1]
    for number in range(len(arr) - 1, 0, -1):  # Inverse range, i.e. right to left in the array
        if to_insert < arr[number - 1]:
            arr[number] = arr[number - 1]
            print(*arr)
        else:
            arr[number] = to_insert
            print(*arr)
            break  # If right position found, break the loop
        if number-2 == -1:  # If arrived at last useful position, replace the first element with number to insert and
            # print the array
            arr[0] = to_insert
            print(*arr)

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)
