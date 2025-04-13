import time
from _1_detection import detect_text
from _1_detection import str2bool
from _2_crop_text_blocks import crop_text_blocks
from _3_recognition import recognition
from _4_translate import translate_text

import argparse
import os
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageTk
import pyautogui
import pynput
import keyboard
import pygetwindow as gw
import tkinter as tk

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

def text_siquences(text_blocks, text_recognititon):
    # Initialize the text_pairs array
    text_pairs = np.empty((len(text_blocks), 4), dtype=object)
    block_keys = list(text_blocks.keys())

    # Populate the text_pairs array
    for i, key in enumerate(block_keys):
        text_pairs[i][0] = key  # Block name
        coords = text_blocks[key]
        text = text_recognititon.get(key, "")
        avg_char_width = (coords[2] - coords[0]) / max(len(text), 1)  # Avoid division by zero
        text_pairs[i][1] = avg_char_width

        # Find left and right pairs
        left_pair = None
        right_pair = None
        for other_key in block_keys:
            if other_key == key:
                continue
            other_coords = text_blocks[other_key]
            if abs(coords[1] - other_coords[1]) < avg_char_width:  # Check if on the same horizontal line
                if other_coords[0] > coords[2] and (other_coords[0] - coords[2]) <= 1.5 * avg_char_width:
                    right_pair = other_key
                elif coords[0] > other_coords[2] and (coords[0] - other_coords[2]) <= 1.5 * avg_char_width:
                    left_pair = other_key
        text_pairs[i][2] = left_pair
        text_pairs[i][3] = right_pair

    # Initialize the text_sequences array
    text_sequences = np.empty((len(text_blocks), 2), dtype=object)

    # Populate the text_sequences array
    visited = set()
    for i, key in enumerate(block_keys):
        if key in visited:
            continue
        sequence = []
        translations = []
        current_key = key

        # Traverse to the left
        while current_key:
            sequence.insert(0, current_key)
            translations.insert(0, text_recognititon.get(current_key, ""))
            visited.add(current_key)
            current_index = np.where(text_pairs[:, 0] == current_key)[0][0]
            current_key = text_pairs[current_index][2]

        # Traverse to the right
        current_key = text_pairs[np.where(text_pairs[:, 0] == key)[0][0]][3]
        while current_key:
            sequence.append(current_key)
            translations.append(text_recognititon.get(current_key, ""))
            visited.add(current_key)
            current_index = np.where(text_pairs[:, 0] == current_key)[0][0]
            current_key = text_pairs[current_index][3]

        text_sequences[i][0] = sequence
        text_sequences[i][1] = translations

    # Initialize the merged_text array
    merged_text = []

    for sequence, translations in text_sequences:
        if sequence is None or translations is None:
            continue
        if not sequence or not translations:
            continue

        # Merge coordinates
        tl_x_min = min(text_blocks[block][0] for block in sequence)
        tl_y_min = min(text_blocks[block][1] for block in sequence)
        tr_x_max = max(text_blocks[block][2] for block in sequence)
        tr_y_min = min(text_blocks[block][3] for block in sequence)
        br_x_max = max(text_blocks[block][4] for block in sequence)
        br_y_max = max(text_blocks[block][5] for block in sequence)
        bl_x_min = min(text_blocks[block][6] for block in sequence)
        bl_y_max = max(text_blocks[block][7] for block in sequence)
        merged_coords = (tl_x_min, tl_y_min, tr_x_max, tr_y_min, br_x_max, br_y_max, bl_x_min, bl_y_max)

        # Merge translations into a single sentence
        merged_sentence = " ".join(translations)

        # Append to the merged_text array
        merged_text.append([merged_coords, merged_sentence])

    # Convert merged_text to a numpy array
    merged_text = np.array(merged_text, dtype=object)
    return text_pairs, text_sequences, merged_text

def draw_quadrangles_on_image(merged_text, input_image_path, output_image_path):
    # Load the image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Unable to load image at {input_image_path}")
        return

    # Iterate through merged_text and draw quadrangles
    for row in merged_text:
        coords = row[0]  # Extract coordinates
        if len(coords) != 8:
            print(f"Error: Invalid coordinates {coords}")
            continue

        # Extract points
        points = np.array([
            [coords[0], coords[1]],  # Top-left
            [coords[2], coords[3]],  # Top-right
            [coords[4], coords[5]],  # Bottom-right
            [coords[6], coords[7]]   # Bottom-left
        ], dtype=np.int32)

        # Draw the quadrangle
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # Save the resulting image
    cv2.imwrite(output_image_path, image)
    print(f"Image with quadrangles saved at {output_image_path}")

def create_mask_from_coordinates(merged_text, input_image_path, output_image_path):
    # Load the original image to get its dimensions
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Unable to load image at {input_image_path}")
        return

    # Create a black image with the same dimensions as the original image
    mask = np.zeros_like(image, dtype=np.uint8)

    # Iterate through merged_text and draw white quadrangles on the mask
    for row in merged_text:
        coords = row[0]  # Extract coordinates
        if len(coords) < 8:
            print(f"Error: Invalid coordinates {coords}")
            continue

        # Extract points
        points = np.array([
            [coords[0], coords[1]],  # Top-left
            [coords[2], coords[3]],  # Top-right
            [coords[4], coords[5]],  # Bottom-right
            [coords[6], coords[7]]   # Bottom-left
        ], dtype=np.int32)

        # Draw the quadrangle as a filled white polygon
        cv2.fillPoly(mask, [points], color=(255, 255, 255))

    # Save the resulting mask
    cv2.imwrite(output_image_path, mask)
    print(f"Mask image saved at {output_image_path}")

def draw_text_on_image(merged_text, input_image_path, output_image_path):
    # Load the image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Unable to load image at {input_image_path}")
        return

    # Iterate through merged_text and draw text in quadrangular areas
    for row in merged_text:
        coords = row[0]  # Extract coordinates
        text = row[1]    # Extract text
        if len(coords) < 8:
            print(f"Error: Invalid coordinates {coords}")
            continue

        # Extract points
        points = np.array([
            [coords[0], coords[1]],  # Top-left
            [coords[2], coords[3]],  # Top-right
            [coords[4], coords[5]],  # Bottom-right
            [coords[6], coords[7]]   # Bottom-left
        ], dtype=np.int32)

        # Calculate the bounding box for the quadrangle
        x_min = min(points[:, 0])
        y_min = min(points[:, 1])
        x_max = max(points[:, 0])
        y_max = max(points[:, 1])

        # Calculate font scale based on the height of the bounding box
        box_height = y_max - y_min
        font_scale = box_height / 30.0  # Adjust the divisor for better scaling
        font_thickness = max(1, int(font_scale))

        # Calculate text size and position
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = x_min
        text_y = y_min + text_size[1]

        # Draw the text on the image
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)

    # Save the resulting image
    cv2.imwrite(output_image_path, image)
    print(f"Image with text saved at {output_image_path}")

def draw_text_in_quadrangles(merged_text, input_image_path, output_image_path):
    # Load the image using PIL
    image = Image.open(input_image_path)
    draw = ImageDraw.Draw(image)

    # Set font to support Unicode characters
    font_path = "C:/Windows/Fonts/arial.ttf"  # Update with a valid font path
    if not os.path.exists(font_path):
        print(f"Error: Font file not found at {font_path}")
        return
    font = ImageFont.truetype(font_path, size=20)  # Default size, will be adjusted dynamically

    # Iterate through merged_text and draw text in quadrangular areas
    for row in merged_text:
        coords = row[0]  # Extract coordinates
        text = row[2]    # Extract text
        if len(coords) != 8:
            print(f"Error: Invalid coordinates {coords}")
            continue

        # Extract points
        points = np.array([
            [coords[0], coords[1]],  # Top-left
            [coords[2], coords[3]],  # Top-right
            [coords[4], coords[5]],  # Bottom-right
            [coords[6], coords[7]]   # Bottom-left
        ], dtype=np.int32)

        # Draw the quadrangle as a filled white polygon
        draw.polygon([tuple(point) for point in points], fill=(255, 255, 255))

        # Calculate the bounding box for the quadrangle
        x_min = min(points[:, 0])
        y_min = min(points[:, 1])
        x_max = max(points[:, 0])
        y_max = max(points[:, 1])

        # Calculate font size based on the height of the bounding box
        box_height = y_max - y_min
        font_size = max(10, box_height // 2)  # Adjust font size dynamically
        font = ImageFont.truetype(font_path, size=font_size)

        # Calculate text size and position
        width = draw.textlength(text, font=font)
        height = font_size;
        text_x = x_min + (x_max - x_min - width) // 2
        text_y = y_min + (y_max - y_min - height) // 2

        # Draw the text on the image
        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))

    # Save the resulting image
    image.save(output_image_path)
    print(f"Image with text in quadrangles saved at {output_image_path}")

def translate_all_siquences(merged_text):
    merged_text_with_translation = []
    #merged_text_with_translation = merged_text.copy()
    for row in merged_text:
        text_to_translate = row[1]  # Extract the text to translate
        text_after_translation = translate_text(text_to_translate)  # Translate the text
        merged_text_with_translation.append([row[0], row[1], text_after_translation])
        print(text_after_translation)
    return merged_text_with_translation

def make_all_steps():
    print("Let's start the process")
        
    if True:
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
        
        text_pairs, text_siquences_var, merged_text = text_siquences(text_blocks, text_recognititon)
        print("Text Pairs:")
        print(text_pairs)
        print("*" * 50)
        print("Text Sequences:")
        print(text_siquences_var)
        print("*" * 50)
        print("Merged Text:")
        print(merged_text)
        print("*" * 50)
        
        merged_text = translate_all_siquences(merged_text)
        
        draw_quadrangles_on_image(
            merged_text=merged_text,
            input_image_path="figures/img_to_translate.jpg",
            output_image_path="result/img_to_translate_features.jpg"
        )
        
        create_mask_from_coordinates(
            merged_text=merged_text,
            input_image_path="figures/img_to_translate.jpg",
            output_image_path="result/img_to_translate_for_text_erase.jpg"
        )
        
        draw_text_in_quadrangles(
            merged_text=merged_text,
            input_image_path="figures/img_to_translate.jpg",
            output_image_path="end_result/img_to_translate_with_text.jpg"
        )
        
        # Clean up files after processing    
        delete_all_files_in_folder('result/blocks')
        delete_all_files_in_folder('result')
        #delete_file('figures/img_to_translate.jpg')
        
        #show_translation(output_folder, file_name)

def show_translation(output_folder, file_name):
    
    file_path = os.path.join(output_folder, file_name)
    
    def on_click(event):
        # Ensure the image exists
        if not os.path.exists(file_path):
            print(f"Image {file_name} not found in {output_folder}")
            exit()
        
        # Create a tkinter window
        root = tk.Tk()
        
        # Delete the image file
        os.remove(file_path)
        print(f"Image {file_name} deleted from {output_folder}")
    
        # Close the tkinter window
        root.destroy()
    
    active_window = gw.getActiveWindow()
    if active_window:
        # Get the window's position and size
        left, top, right, bottom = active_window.left, active_window.top, active_window.right, active_window.bottom
        width, height = right - left, bottom - top
        
        # Take a screenshot of the active window
        #screenshot = pyautogui.screenshot(region=(left, top, width, height))

        # Create a tkinter window
        root = tk.Tk()
        root.title("Image Viewer")

        # Load the image
        image = Image.open(file_path)
        photo = ImageTk.PhotoImage(image)

        # Set the geometry of the tkinter window to match the region
        root.geometry(f"{width}x{height}+{left}+{top}")

        # Make the window stay on top and remove the title bar
        root.attributes("-topmost", True)
        root.overrideredirect(True)

        # Create a label to display the image
        label = tk.Label(root, image=photo)
        label.pack()

        # Bind the click event to the on_click function
        label.bind("<Button-1>", on_click)

        # Run the tkinter main loop
        root.mainloop()
        
        # Simulate a click on the current application
        print('pyautogui.click()')
    else:
        print("No active window found.")

def take_screenshot_on_press_key(output_folder, file_name):

    # Ensure the folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    def take_screenshot():
        
        active_window = gw.getActiveWindow()
        if active_window:
            # Get the window's position and size
            left, top, right, bottom = active_window.left, active_window.top, active_window.right, active_window.bottom
            width, height = right - left, bottom - top
            
            # Take a screenshot of the active window
            screenshot = pyautogui.screenshot(region=(left, top, width, height))

            screenshot.save(os.path.join(output_folder, file_name))
            print(f"Screenshot saved as {file_name} in {output_folder}")
            
            make_all_steps()
            
            show_translation("end_result", "img_to_translate_with_text.jpg")
        else:
            print("No active window found.")

    # Set up the event listener for the mouse scroll button
    keyboard.add_hotkey('ctrl+1', take_screenshot)

    print("Press the mouse scroll button to take a screenshot.")
    keyboard.wait('esc') # Keep the script running until 'esc' is pressed
    
def _take_screenshot_on_press_key(output_folder, file_name):
    
    def on_press(key):
        try:
            # Check if the combination Ctrl+Q is pressed
            if key == pynput.keyboard.Key.ctrl_l or key == pynput.keyboard.Key.ctrl_r:
                on_press.ctrl_pressed = True
            elif key == pynput.keyboard.KeyCode.from_char('1') and on_press.ctrl_pressed:
                # Take a screenshot
                screenshot = pyautogui.screenshot()
                # Ensure the output folder exists
                os.makedirs(output_folder, exist_ok=True)
                # Save the screenshot
                screenshot_path = os.path.join(output_folder, file_name)
                screenshot.save(screenshot_path)
                print(f"Screenshot saved at {screenshot_path}")
                return False  # Stop the listener after taking the screenshot
        except AttributeError:
            pass

    def on_release(key):
        # Reset the Ctrl key state when released
        if key == pynput.keyboard.Key.ctrl_l or key == pynput.keyboard.Key.ctrl_r:
            on_press.ctrl_pressed = False

    # Initialize the Ctrl key state
    on_press.ctrl_pressed = False

    # Start listening for keyboard events
    with pynput.keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        print("Waiting for Ctrl+1 key combination...")
        listener.join()

if __name__ == '__main__':
    
    output_folder = "figures"
    file_name = "img_to_translate.jpg" #png"
    take_screenshot_on_press_key(output_folder, file_name)
    
