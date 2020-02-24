import random
outcomes = {
1: 0,
2: 0,
3: 0,
4: 0,
5: 0,
6: 0,
}

for i in range(10000):
    outcomes[random.randint(1, 6)] += 1
for key, count in sorted(outcomes.items()):
    print(key, count/10000*100)
    
import random
# List of prize categories, to help to print out in order
prizes = [ "Jackpot", "Match5+1", "Match5", "Match4", "Match3", "LuckyDip" ]
# Use sets for collections of balls
entry = set(random.sample(range(1, 60), 6)) # Our entry: 6 balls
# Set up dictionary to count wins
wins = { p: 0 for p in prizes }
# Simulate a year's worth of Lotto draws
for i in range(105):
# Make a draw: 6 main numbers and 1 bonus ball, all drawn at same time
    draw = random.sample(range(1, 60), 7) # Draw: 6 balls + bonus balls
    main = set(draw[:6]) # Separate main balls from bonus (easier coding)
    bonus = draw[6]
# Find number of matching balls
    matched = main & entry # Set intersection: items in both
    matches = len(matched) # Number of matching main balls
# Find which prize (if any)
    if matches == 6:
        wins["Jackpot"] += 1
    elif matches == 5 and bonus in entry:
        wins["Match5+1"] += 1
    elif matches == 5:
        wins["Match5"] += 1
    elif matches == 4:
        wins["Match4"] += 1
    elif matches == 3:
        wins["Match3"] += 1
    elif matches == 2:
        wins["LuckyDip"] += 1
# Print out results (in prize order)
print("LOTTO RESULTS FOR ONE YEAR")
print("==========================")
print("")
for p in prizes:
    print("%-8s : %d" % (p, wins[p]))