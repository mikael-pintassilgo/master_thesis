from _1_detection import detect_text
from _1_detection import str2bool
from _2_crop_text_blocks import crop_text_blocks
from _3_recognition import recognition
import argparse
import os

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='models/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='figures', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()
# Function to delete all files in a folder
def delete_all_files_in_folder(folder_path):
    try:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"File {file_path} deleted successfully.")
    except FileNotFoundError:
        print(f"Folder {folder_path} not found.")
    except Exception as e:
        print(f"Error occurred while deleting files in folder {folder_path}: {e}")

# Function to delete a file
def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"File {file_path} deleted successfully.")
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"Error occurred while deleting file {file_path}: {e}")

if __name__ == '__main__':
    detect_text(args)
    text_blocks = crop_text_blocks()
    text_recognititon = recognition()
    
    print("*" * 50)
    print("Text Blocks:")  
    print(text_blocks)
    print("*" * 50)
    print("Text Recognition:")
    print(text_recognititon)
    print("*" * 50)
    
    # Clean up files after processing    
    delete_all_files_in_folder('result/blocks')
    delete_all_files_in_folder('result')
    #delete_file('figures/img_to_translate.jpg')