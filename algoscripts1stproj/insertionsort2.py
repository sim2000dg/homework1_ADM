
# Simply an extension of insertionsort1 where the whole array is taken
# into consideration for the sorting
def insertionSort2(n, arr):
    i = 2
    while i <= len(arr):
        to_insert = arr[i-1]
        for number in range(i-1, 0, -1):
            if to_insert < arr[number - 1]:
                arr[number] = arr[number - 1]
            else:
                arr[number] = to_insert
                print(*arr)
                break
            if number - 2 == -1:
                arr[0] = to_insert
                print(*arr)
        i += 1


if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)
