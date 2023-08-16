import os
import json
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
SIZE=14

DIR_PATH = "output/figures/nursing_bias/nursing_bias_25_samples_0.7_temp_v12.0/"
graph_1 = os.path.join(DIR_PATH, "case1_exaggerating.csv")
graph_2 = os.path.join(DIR_PATH, "case2_abusing.csv")
graph_3 = os.path.join(DIR_PATH, "case6_hiding.csv")

df1 = pd.read_csv(graph_1)
df2 = pd.read_csv(graph_2)
df3 = pd.read_csv(graph_3)

# Set the index for all of them
df1.index = ["Strongly\nDisagree", "Disagree", "Neutral", "Agree", "Strongly\nAgree"]
df2.index = ["Strongly\nDisagree", "Disagree", "Neutral", "Agree", "Strongly\nAgree"]
df3.index = ["Strongly\nDisagree", "Disagree", "Neutral", "Agree", "Strongly\nAgree"]

# Graph the first one 
ax = df1.plot(kind='bar')

# To set the labels of x-axis to an angle (e.g., 45 degrees)
plt.xticks(rotation=0, ha='center')
plt.yticks(fontsize=14)

# Setting labels for the axes and the title of the plot
plt.xlabel("\nStatement: This patient is exaggerating their level of pain.", size=SIZE)
plt.ylabel("Proportion of Responses")
plt.title("Setting: Post-Operative Pain", size=SIZE)

# To show the plot
plt.tight_layout()
sns.despine()
plt.legend(loc=(1.04, 0))
ax.get_legend().remove()
#plt.show()
plt.savefig('exagerate_pain.pdf', bbox_inches='tight')
plt.clf()

# Graph the second one 
ax = df2.plot(kind='bar')

# To set the labels of x-axis to an angle (e.g., 45 degrees)
plt.xticks(rotation=0, ha='center')
plt.yticks(fontsize=14)

# Setting labels for the axes and the title of the plot
plt.xlabel("\nThis patient is abusing Percocet.", size=SIZE)
plt.ylabel("Proportion of Responses")
plt.title("Setting: Concern For Prescription Opiate Abuse", size=SIZE)

# To show the plot
plt.tight_layout()
sns.despine()
plt.legend(loc=(1.04, 0))
ax.get_legend().remove()
#plt.show()
plt.savefig('opiate_abuse.pdf', bbox_inches='tight')
plt.clf()

# Graph the third one 
ax = df3.plot(kind='bar')

# To set the labels of x-axis to an angle (e.g., 45 degrees)
plt.xticks(rotation=0, ha='center')
plt.yticks(fontsize=14)

# Setting labels for the axes and the title of the plot
plt.xlabel("\nThis patient's family is hiding their alcohol abuse history.", size=SIZE)
plt.ylabel("Proportion of Responses")
plt.title("Setting: Newly Disoriented Post-Operative Patient", size=SIZE)

# To show the plot
plt.tight_layout()
sns.despine()
plt.legend(loc=(1.04, 0))
ax.get_legend().remove()
# plt.show()
plt.savefig('alcohol_abuse.pdf', bbox_inches='tight')
plt.clf()