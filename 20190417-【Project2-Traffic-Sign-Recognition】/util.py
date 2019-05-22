import pickle
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(23)

def load_traffic_sign_data(training_file):
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    X_train, y_train = train['features'], train['labels']

    return X_train, y_train


def show_random_samples(X_train, y_train, n_classes):
    # show a random sample from each class of the traffic sign dataset
    rows, cols = 4, 12
    fig, ax_array = plt.subplots(rows, cols)
    plt.suptitle('Random Samples (one per class)')
    for class_idx, ax in enumerate(ax_array.ravel()):
        if class_idx < n_classes:
            # show a random image of the current class
            cur_X = X_train[y_train == class_idx]
            cur_img = cur_X[np.random.randint(len(cur_X))]
            ax.imshow(cur_img)
            ax.set_title('{:02d}'.format(class_idx))
        else:
            ax.axis('off')
    # hide both x and y ticks
    plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
    plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
    plt.draw()


def show_classes_distribution(n_classes, y_train, n_train):
    # bar-chart of classes distribution
    train_distribution = np.zeros(n_classes)
    for c in range(n_classes):
        train_distribution[c] = np.sum(y_train == c) / n_train
    fig, ax = plt.subplots()
    col_width = 1
    bar_train = ax.bar(np.arange(n_classes), train_distribution, width=col_width, color='r')
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Class Label')
    ax.set_title('Distribution')
    ax.set_xticks(np.arange(0, n_classes, 5) + col_width)
    ax.set_xticklabels(['{:02d}'.format(c) for c in range(0, n_classes, 5)])
    plt.show()


if __name__ == "__main__":
    X_train, y_train = load_traffic_sign_data('./traffic-signs-data/train.p')

    # Number of examples
    n_train = X_train.shape[0]

    # What's the shape of an traffic sign image?
    image_shape = X_train[0].shape

    # How many classes?
    n_classes = np.unique(y_train).shape[0]

    print("训练数据集的数据个数 =", n_train)
    print("图像尺寸  =", image_shape)
    print("类别数量 =", n_classes)

    show_random_samples(X_train, y_train, n_classes)
    show_classes_distribution(n_classes, y_train, n_train)