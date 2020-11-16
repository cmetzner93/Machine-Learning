from collections import defaultdict
import matplotlib.pyplot as plt
import os


# Functions to analyze the data
def read_data_in_dict(filepath):
    # File is expecting to have 7 columns
    # Open file with losses, accuracy, and timestamps
    f = open(filepath, "r")
    prime_dictionary = defaultdict(list)
    for line in f:
        line = line.strip().replace(',', '').replace("[", "").replace("]", "")
        line = line.split()
        for key, element in enumerate(line):
            prime_dictionary[key].append(element)

    # Dictionary prime_dictionary contains values for all batches in each epochs
    # Dictionary adjusted_dict contains the averages of the batches.
    adjusted_dict = defaultdict(list)
    for key in range(0, 7):
        #print("Current key: ", key)
        epoch = 0
        for index, element in enumerate(prime_dictionary.get(key)):
            # Key = 6: Time. Take sum of all batches and add to dict once epoch is completed
            if key == 6:
                epoch += float(element)
                if (index + 1) % 12 == 0:
                    adjusted_dict[key].append(round(epoch, 0))
            # Other keys are losses and accuracies
            else:
                # Take the loss and accuracy of the final batch in epoch and append element to the dict
                if (index + 1) % 12 == 0:
                    adjusted_dict[key].append(float(element))
        #print(adjusted_dict[key])
    return adjusted_dict


# Function to plot the losses based on diagnostic files
# Due to position of key in dictionary from function read_data_in_dict information is hardcoded
def plot_losses(curr_dict, name):
    color = ['blue', 'orange', 'green']
    # Losses
    fig = plt.figure()
    fig.suptitle('Loss vs Time', fontsize=20)
    plt.xlabel('Time progressed in [s]', fontsize=18)
    plt.ylabel('Loss', fontsize=16)
    plt.xlim(0, 5500)
    max_val_0 = max(curr_dict[0])
    max_val_2 = max(curr_dict[2])
    max_val_4 = max(curr_dict[4])
    max_val = max(max_val_0, max_val_2, max_val_4)
    plt.ylim(0, (max_val+max_val*0.1))
    plt.plot(curr_dict[6], curr_dict[0], color=color[0], label='Real_Loss_Disc')
    plt.plot(curr_dict[6], curr_dict[2], color=color[1], label='Fake_Loss_Disc')
    plt.plot(curr_dict[6], curr_dict[4], color=color[2], label='Loss_Gen')
    plt.legend()
    plt.savefig(name + '_loss_vs_epoch')
    plt.close('all')


# Function to plot the accuracies based on diagnostic files
# Positions are hard coded
def plot_acc(curr_dict, name):
    color = ['blue', 'orange', 'green']
    # Accuracies
    fig = plt.figure()
    fig.suptitle('Accuracy vs Time', fontsize=20)
    plt.xlabel('Time progressed in [s]', fontsize=18)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlim(0, 5500)
    plt.ylim(0, 1.1)
    plt.plot(curr_dict[6], curr_dict[1], color=color[0], label='Real_Acc_Disc')
    plt.plot(curr_dict[6], curr_dict[3], color=color[1], label='Fake_Acc_Disc')
    plt.plot(curr_dict[6], curr_dict[5], color=color[2], label='Acc_Gen')
    plt.legend()
    plt.savefig(name + '_acc_vs_epoch')
    plt.close('all')


# Main function which calls functions
# read_data_in_dict, plot_losses, plot_acc
def main():
    # path to where the file is located
    path = os.path.join("C:", "\\Users", "chris", "Desktop", "tests")
    diagnostic_file = '_cDCGAN_diagnostics.txt'
    file_path = os.path.join(path, test, diagnostic_file)
    current_dict = read_data_in_dict(filepath=file_path)
    plot_losses(curr_dict=current_dict, name=path+'\\'+test)
    plot_acc(curr_dict=current_dict, name=path+'\\'+test)


if __name__ == '__main__':
    main()
