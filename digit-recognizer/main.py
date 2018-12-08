from digitrecognizer.dataset import Mnist, mnist_dataset

if __name__ == '__main__':
    train_dataset, eval_dataset = mnist_dataset('input/train.csv', 0.1)

    for sample in train_dataset:
        print(sample)
