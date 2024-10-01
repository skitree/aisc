import torch as ch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the seaborn style to 'whitegrid' for a clean look with grid lines
sns.set(style='white', context='paper', palette='muted')

# Increase default font size for better readability
plt.rcParams.update({'font.size': 12})

# Make the font times new roman
plt.rcParams['font.family'] = 'Times New Roman'

# Load the cos_sims_good.pt file
cos_sims = ch.load("cos_sims.pt")

# Plot the cosine similarities with improved aesthetics
plt.figure(figsize=(5, 3))
for i, (lower, upper) in cos_sims.items():
    # Use a more subtle color, like blue, with markers for clarity
    plt.plot([lower, upper], [6-i, 6-i], marker='o', linestyle='-', color='blue')
    # Add a marker for the mean
    plt.plot(cos_sims[i].mean(), 6-i, marker='|', markersize=5, color='gray')
    # Annotation for the mean
    plt.text(cos_sims[i].mean(), 6.3-i, f'{cos_sims[i].mean():.2f}', ha='center', va='center', color='gray')

# Customizing the plot
plt.yticks(list(cos_sims.keys()), [f'{6-i}' for i in cos_sims])
plt.xticks([.8, .85, .9, .95, 1], ['0.8', '0.85', '0.9', '0.95', '1'])
plt.title('Cosine Similarity of Explanations (95% CI)')
plt.ylabel('Number of Supply Chain Nodes')
plt.xlabel('Cosine Similarity')
plt.ylim(0.5, 5.8)
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Optional: Add a legend if needed
# plt.legend(title='Nodes', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()  # Adjust the layout to make room for the elements
plt.savefig('cosine_similarity.pdf', bbox_inches='tight')

# Data
categories = ['Individuals with no recourse (%)', 'Distance to recourse']
true_recourse = [1.4, 1.326]
estimated_recourse = [2.5, 2.534]

x = np.arange(len(categories))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(5, 3))
rects1 = ax.bar(x - width/2, true_recourse, width, label='True Recourse', color='blue')
rects2 = ax.bar(x + width/2, estimated_recourse, width, label='Estimated Recourse', color='orange')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Values')
ax.set_title('Comparison of True and Estimated Recourse')
ax.set_xticks(x)
ax.set_ylim(0, 4)
ax.set_xticklabels(categories)
ax.legend()

# Function to auto-label the bars with their values
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('recourse.pdf', bbox_inches='tight')