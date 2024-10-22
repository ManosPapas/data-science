import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Set Seaborn style
sns.set(style='ticks', color_codes=True)

# Universal visualization setup
plt.style.use('dark_background')  # Set the background to dark
sns.set(style='darkgrid', palette='muted')  # Set seaborn style and palette

# Customizing font sizes and colors for all plots
plt.rcParams['axes.titlesize'] = 20  # Title font size
plt.rcParams['axes.labelsize'] = 16  # Label font size
plt.rcParams['xtick.labelsize'] = 14  # X tick label size
plt.rcParams['ytick.labelsize'] = 14  # Y tick label size
plt.rcParams['text.color'] = 'white'  # Set text color to white
plt.rcParams['axes.labelcolor'] = 'white'  # Set axes label color to white
plt.rcParams['axes.titleweight'] = 'bold'  # Bold title
