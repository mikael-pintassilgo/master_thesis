import os
import time
import tkinter as tk
from PIL import Image, ImageTk
import pygetwindow as gw
import pyautogui

def show_translation(output_folder, file_name):
    
    file_path = os.path.join(output_folder, file_name)
    
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

        # Define a function to close the window on click
        def close_window(event):
            root.destroy()

        # Bind the click event to the close_window function
        label.bind("<Button-1>", close_window)

        # Run the tkinter main loop
        root.mainloop()
        
        # Simulate a click on the current application
        print('pyautogui.click()')
        
        # Click at the current mouse position without moving the mouse
        pyautogui.click()
    else:
        print("No active window found.")

time.sleep(3)  # Wait for the screenshot to be taken
show_translation("end_result", "img_to_translate_with_text.jpg")