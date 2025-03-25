import matplotlib.pyplot as plt


def main():
    accuracy_list = [90.93, 91, 91, 90.77, 90.09, 93.21]
    score_list = [0.2153, 0.1937, 0.1697, 0.1896, 0.11, 0.1559]
    name_list = [
        "griezmann",
        "dembele",
        "unstructured_pruning_dembele",
        "unstructured_pruning_griezmann",
        "deschamps",
        "zidane_8bits"
    ]

    plt.scatter(score_list, accuracy_list)
    for i, txt in enumerate(name_list):
        plt.annotate(txt, (score_list[i], accuracy_list[i]))

    plt.xlabel("Score")
    plt.ylabel("Accuracy")
    plt.title("Score vs Accuracy")
    plt.savefig("results/score_accuracy.png")


if __name__ == "__main__":
    main()
