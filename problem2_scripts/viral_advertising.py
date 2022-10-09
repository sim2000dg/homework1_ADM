import os

# Three variables are needed to check the progress:
# - "likes_stage" to check the number of new likes from the current iteration
# - "reached" to propagate to the next iteration the number of
# new people reached at the current iteration
# - "likes" to count the number of likes overall
# Once this is clear, the rest is trivial

def viralAdvertising(n):
    reached = 5
    likes = 0
    for _ in range(n):
        likes_stage = reached//2
        likes += likes_stage
        reached = likes_stage*3
    return likes

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()