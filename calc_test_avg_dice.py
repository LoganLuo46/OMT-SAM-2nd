import re
import csv
import matplotlib.pyplot as plt

log_file = "Zero_shot_7th_round.log"
out_file = "Zero_shot_7th_round.csv"
plot_file = "Zero_shot_7th_round.png"

# List of organs (update if your gan set changes)
organ_names = [
    "Right_kidney", "Ivc", "Gallbladder", "Esophagus", 
    "Left_kidney", "Left_adrenal_gland", "Aorta", "Stomach"
]
EXPECTED_ORGAN_COUNT = len(organ_names)

epoch_pattern = re.compile(r"Epoch (\d+)/\d+")
dice_pattern = re.compile(r"[\w_]+ Validation Complete: Loss: [\d\.]+, Dice: ([\d\.]+)")

results = []
epoch = None
epoch_dices = []
first_epoch = None
first_epoch_dices = []

with open(log_file, "r") as f:
    lines = f.readlines()

for line in lines:
    epoch_match = epoch_pattern.match(line)
    if epoch_match:
        # On new epoch, save previous epoch's result if full set of dice values
        if epoch is not None and len(epoch_dices) == EXPECTED_ORGAN_COUNT:
            results.append((epoch, sum(epoch_dices) / len(epoch_dices)))
        # For the very first epoch, store its number and start collecting dice
        if first_epoch is None:
            first_epoch = int(epoch_match.group(1))
            first_epoch_dices = []
        epoch = int(epoch_match.group(1))
        epoch_dices = []
    else:
        dice_match = dice_pattern.match(line.strip())
        if dice_match:
            dice = float(dice_match.group(1))
            epoch_dices.append(dice)
            if first_epoch is not None and epoch == first_epoch:
                first_epoch_dices.append(dice)

# After all lines, save the first epoch if it has a full set of dice values
if first_epoch is not None and len(first_epoch_dices) == EXPECTED_ORGAN_COUNT:
    # Only add if not already in results
    if not results or results[0][0] != first_epoch:
        results.insert(0, (first_epoch, sum(first_epoch_dices) / len(first_epoch_dices)))

# Save the last epoch only if it has a full set of dice values
if epoch is not None and len(epoch_dices) == EXPECTED_ORGAN_COUNT:
    # Avoid duplicate if last epoch is already in results
    if not results or results[-1][0] != epoch:
        results.append((epoch, sum(epoch_dices) / len(epoch_dices)))

# Write to csv file (with header)
title = ["epoch", "avg_dice"]
with open(out_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(title)
    for epoch, avg_dice in results:
        writer.writerow([epoch, f"{avg_dice:.4f}"])

# Print a nice table to the console
print("\n{:^10} | {:^10}".format("Epoch", "Avg Dice"))
print("-"*24)
for epoch, avg_dice in results:
    print("{:^10} | {:^10.4f}".format(epoch, avg_dice))

# Plot the curve
epochs = [e for e, _ in results]
avg_dices = [d for _, d in results]
plt.figure(figsize=(10,6))
plt.plot(epochs, avg_dices, marker='o', color='b', label='Avg Dice')
plt.xlabel('Epoch')
plt.ylabel('Average Dice')
plt.title('Average Dice per Epoch')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(plot_file)
print(f"\nGenerated {out_file} (table) and {plot_file} (curve plot)!") 