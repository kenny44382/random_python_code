import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def add_points(x, y1, y2):
    #x,y1,y2 are all single values
    #number of points
    num_step = int(np.abs(y1 - y2))

    #length of 2 points
    dis = 2
    
    new_x = []
    new_y = []
    
    for i in range(1,num_step,dis):
        new_x.append(x)
        new_y.append(max(y1,y2)-i)
    return new_x,new_y
    
def get_new_x(row):
    x,y = add_points(row['x'], row['y1'],row['y2'])
    return x
def get_new_y(row):
    x,y = add_points(row['x'], row['y1'],row['y2'])
    return y

def flatten_values(a):
    #x = df['x'].values.flatten().tolist()
    x = a.values.flatten().tolist()
    new_x = [item for sublist in x for item in sublist]
    return new_x

data = {
    'x': np.arange(1,150,15), # Sorted x for clearer visualization
}

# Generate y1 and y2 values based on x
data['y1'] = 0 * data['x']
data['y2'] = 1 * data['x']

#data['y3'] = data['x'] * random.choice([1, -1])
#data['y4'] = 2 * data['x'] * random.choice([1, -1]) 

df = pd.DataFrame(data)
df['y5'] = np.random.choice([1, -1], size=len(df))



#filtered 
#df = df[df['y5'] == 1]

distance = 5 #distance filled

df['new_x'] = df.apply(get_new_x, axis = 1)
df['new_y'] = df.apply(get_new_y, axis = 1)

new_x1 = flatten_values(df['new_x'])
new_y1 = flatten_values(df['new_y'])

df.to_csv('data_1.csv')
# 3. Scatter plot all three sets of data
plt.figure(figsize=(10, 7))

# Plot df['x'] versus df['y1']
plt.scatter(df['x'], df['y1'], color='blue', label='Y1 Data', alpha=0.7, s=50)

# Plot df['x'] versus df['y2']
plt.scatter(df['x'], df['y2'], color='red', label='Y2 Data', alpha=0.7, s=50)
# Plot the inserted points (df['new_x'] versus df['new_y'])
plt.scatter(new_x1, new_y1, color='green', label='Y insert data', alpha=0.7, s=50)


# Add titles and labels
plt.title('Scatter Plot: X vs Y1, Y2 with Inserted Points Between')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout() # Adjust layout to prevent labels from overlapping

# Display the plot (optional, won't show in environments without display)
plt.show()