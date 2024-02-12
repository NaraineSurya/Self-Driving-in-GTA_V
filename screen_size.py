import tkinter as tk

# This code provides your screen size in the bounding box value

def get_screen_size():
    root = tk.Tk()
    # Hide the root window
    root.withdraw()
    # Get the screen width and height
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    # Destroy the root window
    root.destroy()
    return {'top': 0, 'left': 0, 'width': width, 'height': height}

bounding_box = get_screen_size()
print(bounding_box)
