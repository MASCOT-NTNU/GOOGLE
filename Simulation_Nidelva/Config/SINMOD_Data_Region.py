import numpy as np
import pandas as pd


box = np.array([[63.4441527, 10.3296626],
                [63.4761121, 10.3948786],
                [63.4528538, 10.45186239],
                [63.4209213, 10.38662725]])
df = pd.DataFrame(box, columns=['lat', 'lon'])
df.to_csv("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Config/SINMOD_Data_Region.csv", index=False)

