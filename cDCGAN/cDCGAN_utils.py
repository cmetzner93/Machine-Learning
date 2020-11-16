import matplotlib.pyplot as plt


# Function to generate and save images with new test data for trained model
def generate_and_save_images(model, epoch, test_input, test_labels, name):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(inputs=[test_input, test_labels], training=False)

    fig = plt.figure(figsize=(4, 5))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 5, i + 1)
        plt.imshow((predictions[i, :, :, 0] + 127.5) * 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(name+'_image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()
    plt.close('all')


# Function store metrics in text file
# add new line after each batch
def save_diagnostics_to_file(name_of_file, diagnostics):
    with open(name_of_file+'.txt', 'w') as f:
        f.writelines("%s\n" % batch for epoch in diagnostics for batch in epoch)
    f.close()
