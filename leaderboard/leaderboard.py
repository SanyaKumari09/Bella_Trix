import csv

with open("leaderboard/leaderboard.csv") as f:
    reader = csv.reader(f)

    print("\nLeaderboard\n")

    for row in reader:
        print(row)