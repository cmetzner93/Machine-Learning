import os
import sys
import shutil
import glob

info_path = r"C:\Users\chris\Desktop\Studium\PhD\Courses\Spring 2020\COSC 525 - Deep Learning\DeepLearning_FinalProject\Dataset\Complete\image_info.txt"


# path to all images
all_data_set_path = r"C:\Users\chris\Desktop\Studium\PhD\Courses\Spring 2020\COSC 525 - Deep Learning\DeepLearning_FinalProject\Dataset\Complete\\"

# path to save benign images
cleaned_dataset_path_benign = r"C:\Users\chris\Desktop\Studium\PhD\Courses\Spring 2020\COSC 525 - Deep Learning\DeepLearning_FinalProject\Dataset\Benign\\"

# path to save malignant images
cleaned_dataset_path_malignant = r"C:\Users\chris\Desktop\Studium\PhD\Courses\Spring 2020\COSC 525 - Deep Learning\DeepLearning_FinalProject\Dataset\Malignant\\"

# path to save normal images
cleaned_dataset_path_normal = r"C:\Users\chris\Desktop\Studium\PhD\Courses\Spring 2020\COSC 525 - Deep Learning\DeepLearning_FinalProject\Dataset\Normal\\"




# open 'filenames.txt'
# filename.txt is created using the information about the
# images from the MIAS database
print('Processing...')
col_num = 2
delimiter = " "
with open(info_path, 'r') as f:
    for line in f:
        col_content = []
        col_content.append(line.split(delimiter)[2])
        if col_content[0] == "NORM":
            # source file from "complete"  directory
            src = all_data_set_path + str(line.split(' ')[0]) + '.pgm'

            # destination file for normal images
            dst = cleaned_dataset_path_normal + str(line.split(' ')[0]) + '_normal.pgm'
            # copy normal images to 'dataset/normal/'
            shutil.copy2(src, dst)

        else:
            col_content = []
            col_content.append(line.split(delimiter)[3])
            if col_content[0] == "B":
                src = all_data_set_path + str(line.split(' ')[0]) + '.pgm'

                # destination file for benign images
                dst = cleaned_dataset_path_benign + str(line.split(' ')[0]) + '_benign.pgm'
                # copy normal images to 'dataset/benign/'
                shutil.copy2(src, dst)
            elif col_content[0] == "M":
                src = all_data_set_path + str(line.split(' ')[0]) + '.pgm'

                # destination file for malignant images
                dst = cleaned_dataset_path_malignant + str(line.split(' ')[0]) + '_malignant.pgm'
                # copy normal images to 'dataset/malignant/'
                shutil.copy2(src, dst)
print('Finished processing!')


from PIL import Image
import glob

print("Generating JPG images")
folders = [cleaned_dataset_path_benign, cleaned_dataset_path_malignant, cleaned_dataset_path_normal]
for folder_path in folders:
    pgm_image_list = glob.glob(folder_path + '*.pgm')
    for filename in pgm_image_list:
        img = Image.open(filename)
        img.save(filename[:-4] + '.jpg')
print("Finished generating!")