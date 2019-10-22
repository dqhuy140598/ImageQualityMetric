import matplotlib.pyplot as plt

def plot_image(image,title):
    plt.imshow(image,cmap='binary')
    plt.title(title)
    plt.show()