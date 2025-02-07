import numpy as np
import matplotlib.pyplot as plt

'''
# Creating a 1D array (like a list)
arr = np.array([1, 20, 19, 21, 4])
print(arr) # Output :[1, 20, 19, 21, 4]

# Creating a 2D array (like a matrix)
TDArray = np.array([[1, 2, 3], [4, 5, 6]])
print(TDArray) # Output : [ [1 2 3][4 5 6] ]

arr1 = np.array([10, 20, 30, 40, 50])
# Getting elements from index 1 to 3 (not including index 3)
arr1 = arr1[1:4]
print(arr1) # Output: [20 30 40]



arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# Get the first two rows, and the first two columns
arr = arr_2d[:2, :2]
print(arr) # Output : [[1 2] [4 5]]


arr = np.array([10, 20, 30, 40, 50])
# Get the last 3 elements
Arr1 = arr[-3:]
print(Arr1) # Output: [30 40 50]


# Sample data
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]
# Creating a line plot
plt.plot(x, y)
# Display the plot
plt.show()


readable: plt.plot(x, y)
plt.title('Simple Line Plot')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.grid(True)
plt.show()


categories = ['A', 'B', 'C', 'D']
values = [5, 7, 3, 8]
plt.bar(categories, values)
plt.title('Bar Chart Example')
plt.xlabel('Category')
plt.ylabel('Value')
plt.show()




group_A = [12, 15, 14, 13, 16, 18, 19, 15, 14, 20, 17, 14, 15,40,45,50,62]
group_B = [12, 17, 15, 13, 19, 20, 21, 18, 17, 16, 15, 14, 16, 15]

plt.boxplot(group_A)
plt.title('group A boxplot example ')
plt.xlabel('Category')
plt.ylabel('Value')
t.show()

# Convert to Grayscale
gray_img = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])

# Plot Grayscale Image
plt.figure(figsize=(5, 5))
plt.imshow(gray_img, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()
plt.show()

plt.boxplot(group_B)
plt.title('group B boxplot example ')
plt.xlabel('Category')
plt.ylabel('Value')

plt.show()


group_A = [12, 15, 14, 13, 16, 18, 19, 15, 14, 20, 17, 14, 15,40,45,50,62]
group_B = [12, 17, 15, 13, 19, 20, 21, 18, 17, 16, 15, 14, 16, 15]

# subPlot 1: Group A
x = np.arange(len(group_A))
y = np.array(group_A)
plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title('Box Plot for Group A')
plt.xlabel('Index')
plt.ylabel('Measurement Values')


# subPlot 2: Group B
x = np.arange(len(group_B))
y = np.array(group_B)
plt.subplot(1, 2, 2)
plt.plot(x, y)
plt.title('Box Plot for Group B')
plt.xlabel('Index')
plt.ylabel('Measurement Values')

# Display the plots
plt.show()



#question 2 

with open("genome.txt", "r") as file:
    genome_sequence = list(file.read().strip()) 
genome_length = len(genome_sequence)

t = np.linspace(0, 4 * np.pi, genome_length) 
x = np.cos(t)
y = np.sin(t)
z = np.linspace(0, 5, genome_length)  

color_map = {'A': 'red', 'T': 'blue', 'C': 'green', 'G': 'yellow'}
colors = [color_map.get(base, 'black') for base in genome_sequence]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=colors, marker='o')

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("3D Helix Representation of Genome Sequence")

plt.show()



#question 3 
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio

image_url = "https://t3.ftcdn.net/jpg/04/54/94/56/360_F_454945621_Fmy7wtnR8cCc99ui7JiZ7tjgESDAIs3r.jpg"
img_array = iio.imread(image_url)


plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(img_array)
plt.title("Original Image")
plt.axis("off")

rotated_img = np.rot90(img_array)
flipped_img = np.fliplr(img_array)

plt.subplot(1, 3, 2)
plt.imshow(rotated_img)
plt.title("Rotated Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(flipped_img)
plt.title("Flipped Image")
plt.axis("off")

plt.show()

gray_img = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])

plt.figure(figsize=(5, 5))
plt.imshow(gray_img, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

'''
#question 4 
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
iris = load_iris()

X = np.array(iris.data)

Y = np.array(iris.target)

mean_values = np.mean(X, axis=0)
median_values = np.median(X, axis=0)
std_values = np.std(X, axis=0)
min_values = np.min(X, axis=0)
max_values = np.max(X, axis=0)

print("Mean values of each feature:", mean_values)
print("Median values of each feature:", median_values)
print("Standard deviation of each feature:", std_values)
print("Minimum values of each feature:", min_values)
print("Maximum values of each feature:", max_values)


sepal_data = X[:, :2]  
plt.figure(figsize=(8, 6))
plt.scatter(sepal_data[:, 0], sepal_data[:, 1], c=Y, cmap='viridis')
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.colorbar(label='Species')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(X[:, 0], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(X[:, 2], X[:, 3], 'go-', label='Petal Length vs Petal Width')
plt.title('Petal Length vs Petal Width')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.grid(True)
plt.legend()
plt.show()


