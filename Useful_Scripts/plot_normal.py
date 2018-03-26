import matplotlib.pyplot as plt 
from extract import extract_actual_alternate

# Add a figure
fig = plt.figure()
ax = fig.add_subplot(111)

# Labels and Title
cycles = 1
y = extract_actual_alternate(filename='../new_old_ratio.txt', cycles=cycles)
x = vector_size = [(i+1)*cycles for i in range(len(y))]

# Plot
ax.plot(x, y, 'g', label='New/Old')
# ax.plot(x, y2[:11], 'g', label='Without BN')
# ax.legend(loc=2)

# Title
name = 'New_Old_ratio'
ax.set_title(name)
ax.set_xlabel('Steps')
ax.set_ylabel('Ratio')

# Save the figure
plt.savefig(name+'.png')