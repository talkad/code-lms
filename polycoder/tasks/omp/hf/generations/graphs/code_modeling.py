
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))


# # CodeBleu
# gpt = []
# poly = []
# comp_bpe = []
# comp_replaced_bpe = []
# comp_tokom = []

# CodeBleu
gpt = [0.422, 0.564, 0.596]
poly = [0.422, 0.588, 0.766]
comp_bpe = [0.543, 0.730, 0.837]
comp_replaced_bpe = [0.638, 0.776, 0.852]
comp_tokom = [0.257, 0.447, 0.596]

gpt = [0.437, 0.579, 0.558]
poly = [0.448, 0.655, 0.796]
comp_bpe = [0.615, 0.772, 0.854]
comp_replaced_bpe = [0.619, 0.755, 0.852]
comp_tokom = [0.248, 0.463, 0.618]


bars = [('comp_tokom', comp_tokom), ('gpt', gpt), 
        ('poly', poly), ('comp_bpe', comp_bpe), ('comp_replaced_bpe', comp_replaced_bpe)]


# Plot the first subplot
xs = ['context-100', 'context-300', 'context-600']

# Plot the second subplot with multiple bars
width = 0.1  # Width of each bar
x2 = np.arange(3)  # X-axis positions for each set of bars

for i, (label, row) in enumerate(bars):
    axs[0].bar(x2 + i * width, row, width=width, label=label)

axs[0].set_title('CodeBleu')
axs[0].set_xlabel('contexts')
axs[0].set_ylabel('CodeBleu score')
axs[0].set_xticks(x2 + (len(bars) - 1) * width / 2)
axs[0].set_xticklabels(xs)
axs[0].legend()


# CodeBERTScore
gpt = [0.813, 0.844, 0.919]
poly = [0.795, 0.888, 0.939]
comp_bpe = [0.773, 0.887, 0.943]
comp_replaced_bpe = [0.805, 0.913, 0.955]
comp_tokom = [0.724, 0.830,  0.889]

gpt = [0.819, 0.887, 0.915]
poly = [0.811, 0.902, 0.949]
comp_bpe = [0.802, 0.903, 0.952]
comp_replaced_bpe = [0.909, 0.913, 0.953]
comp_tokom = [0.731, 0.837,  0.891]

bars = [('comp_tokom', comp_tokom), ('gpt', gpt), 
        ('poly', poly), ('comp_bpe', comp_bpe), ('comp_replaced_bpe', comp_replaced_bpe)]

for i, (label, row) in enumerate(bars):
    axs[1].bar(x2 + i * width, row, width=width, label=label)

axs[1].set_title('CodeBERT Score')
axs[1].set_xlabel('contexts')
axs[1].set_ylabel('CodeBERT score')
axs[1].set_xticks(x2 + (len(bars) - 1) * width / 2)
axs[1].set_xticklabels(xs)
axs[1].legend()


axs[0].grid()
axs[1].grid()

# Adjust layout for better visibility
plt.tight_layout()

# Show the plot
plt.show()

