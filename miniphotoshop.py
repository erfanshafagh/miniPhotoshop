import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
from PIL import Image, ImageTk
import struct
import numpy as np


class MiniPhotoshop:
    def __init__(self, master):
        # Set up the main window
        self.master = master
        self.master.title("Mini-Photoshop")
        self.wait_window = None

        # Create canvas
        self.canvas_width = 706 * 2
        self.canvas_height = 576
        self.master.geometry(f"{self.canvas_width}x{self.canvas_height}")
        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg="lightgray")
        self.canvas.pack()

        # Create menu
        self.menu_bar = tk.Menu(self.master)
        self.master.config(menu=self.menu_bar)

        # Create core operations menu
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Core Operations", menu=self.file_menu)
        self.file_menu.add_command(label="Open File", command=self.open_bmp_file)
        self.file_menu.add_command(label="Grayscale", command=self.grayscale)
        self.file_menu.add_command(label="Ordered Dithering", command=self.ordered_dithering)
        self.file_menu.add_command(label="Auto Level", command=self.auto_level)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.master.quit)

        # Create optional operations menu
        self.file_menu2 = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Optional Operations", menu=self.file_menu2)
        self.file_menu2.add_command(label="Blur", command=self.blur)
        self.file_menu2.add_command(label="Sharpening", command=self.sharpening)
        self.file_menu2.add_command(label="Edge Detection", command=self.edge_detection)
        self.file_menu2.add_separator()
        self.file_menu2.add_command(label="Save the image", command=self.save_image)

        self.edited_image = None
        self.photo = None
        self.original_image = None


    # Core Operations:
    def open_bmp_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("BMP Files", "*.bmp")])
        if file_path:
            try:
                self.read_bmp_info(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open the file: {str(e)}")


    def read_bmp_info(self, file_path):
        with open(file_path, 'rb') as f:
            # Read BMP header
            header = f.read(54)
            if len(header) != 54:
                raise ValueError("Invalid BMP file")

            # Unpack header information
            (signature, file_size, reserved, data_offset,
             header_size, width, height, planes, bpp) = struct.unpack('<2sIIIIIIHH', header[:30])

            if signature != b'BM':
                raise ValueError("Not a BMP file")
            if bpp != 24:
                raise ValueError("Only 24-bit BMP files are supported")
            if width > 704 or height > 576:
                raise ValueError("Image size exceeds 704x576")

        # Read pixel data
        with open(file_path, 'rb') as f:
            f.seek(data_offset)
            pixel_data = f.read()

        # Create image from pixel data
        image = Image.frombytes('RGB', (width, height), pixel_data, 'raw', 'BGR')
        
        # Flip the image vertically (BMP stores rows from bottom to top)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        self.original_image = image
        self.display_image(image)


    def grayscale(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please open an image first")
            return

        # Create a "Please wait" message box
        self.wait()

        # Create an array for the gray image
        tempGray = np.array(self.original_image)

        # Iterate over each pixel in the image
        for x in range(tempGray.shape[0]):
            for y in range(tempGray.shape[1]):
                r, g, b = tempGray[x, y]
                # Convert to grayscale
                gray = int(0.299 * r + 0.587 * g + 0.114 * b)
                tempGray[x, y] = gray, gray, gray

        gray_image = Image.fromarray(tempGray[:, :, 0].astype('uint8'))

        # Destroy the "Please wait" message box
        self.end_wait()

        # Display original and grayscale side by side
        self.display_side_by_side(self.original_image, gray_image)


    def ordered_dithering(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please open an image first")
            return

        # Create a "Please wait" message box
        self.wait()

        # Create an array for the gray image
        tempGray = np.array(self.original_image)

        # Iterate over each pixel in the image
        for x in range(tempGray.shape[0]):
            for y in range(tempGray.shape[1]):
                r, g, b = tempGray[x, y]
                # Convert to grayscale
                gray = int(0.299 * r + 0.587 * g + 0.114 * b)
                tempGray[x, y] = gray, gray, gray

        gray_image = tempGray[:, :, 0]
        
        # 4x4 Bayer matrix
        bayer_matrix = np.array([
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5]
        ]) / 16.0

        # Tile the dither matrix to match image size
        h, w = gray_image.shape
        bayer_tiled = np.tile(bayer_matrix, ((h + 3) // 4, (w + 3) // 4))[:h, :w]
        
        # Apply dithering
        dithered = np.where(gray_image / 255.0 > bayer_tiled, 255, 0).astype(np.uint8)
        
        # Convert back to PIL Image
        dithered_image = Image.fromarray(dithered.astype('uint8'))
        gray_image = Image.fromarray(gray_image.astype('uint8'))

        # Destroy the "Please wait" message box
        self.end_wait()

        # Display original, grayscale, and dithered side by side
        self.display_side_by_side(gray_image, dithered_image)


    def auto_level(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please open an image first")
            return

        # Create a "Please wait" message box
        self.wait()

        # Convert image to numpy array
        img_array = np.array(self.original_image)
        height, width, _ = img_array.shape

        # Process each channel
        for channel in range(3):
            # Calculate histogram
            hist = [0] * 256
            for i in range(height):
                for j in range(width):
                    pixel_value = img_array[i, j, channel]
                    hist[pixel_value] += 1

            # Calculate cumulative distribution function (CDF)
            cdf = [0] * 256
            cdf[0] = hist[0]
            for i in range(1, 256):
                cdf[i] = cdf[i - 1] + hist[i]

            # Find min and max of CDF for normalization
            cdf_min = float('inf')
            cdf_max = float('-inf')
            for value in cdf:
                if value < cdf_min and value != 0:
                    cdf_min = value
                if value > cdf_max:
                    cdf_max = value

            # Create mapping function
            mapping = [0] * 256
            for i in range(256):
                if cdf[i] == 0:
                    mapping[i] = 0
                else:
                    # Scale to 0-255 range
                    mapping[i] = round((cdf[i] - cdf_min) * 255 / (cdf_max - cdf_min))

            # Apply mapping to image
            for i in range(height):
                for j in range(width):
                    pixel_value = img_array[i, j, channel]
                    img_array[i, j, channel] = mapping[pixel_value]

        # Convert back to PIL Image
        auto_leveled = Image.fromarray(img_array.astype('uint8'))

        # Destroy the "Please wait" message box
        self.end_wait()

        # Display original and auto-leveled version side by side
        self.display_side_by_side(self.original_image, auto_leveled)


    # Optional Operations:
    def blur(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please open an image first")
            return

        # Create a "Please wait" message box
        self.wait()

        # Convert image to numpy array
        image_array = np.array(self.original_image)
        height, width, channels = image_array.shape

        # Create a new empty array for the blurred image
        blurred_array = np.zeros_like(image_array)

        # Define the size of the blur kernel 6x6
        kernel_size = 6
        offset = kernel_size // 2

        # Iterate over each pixel in the image
        for y in range(offset, height - offset):
            for x in range(offset, width - offset):
                # For each pixel, average the values of the neighboring pixels
                for c in range(channels):
                    blurred_array[y, x, c] = np.mean(
                        image_array[y - offset:y + offset + 1, x - offset:x + offset + 1, c]
                    )

        # Convert the blurred array back to an image
        blurred_image = Image.fromarray(blurred_array.astype('uint8'))

        # Destroy the "Please wait" message box
        self.end_wait()

        # Display the original and blurred images side by side
        self.display_side_by_side(self.original_image, blurred_image)


    def sharpening(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please open an image first")
            return

        # Create a "Please wait" message box
        self.wait()

        # Convert image to numpy array
        image_array = np.array(self.original_image)
        height, width, channels = image_array.shape

        # Create a new empty array for the sharpened image
        sharpened_array = np.zeros_like(image_array)

        # Define the sharpening kernel
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        kernel_size = 3
        offset = kernel_size // 2

        # Iterate over each pixel in the image
        for y in range(offset, height - offset):
            for x in range(offset, width - offset):
                for c in range(channels):
                    # Apply the sharpening kernel
                    region = image_array[y - offset:y + offset + 1, x - offset:x + offset + 1, c]
                    sharpened_array[y, x, c] = np.clip(np.sum(region * kernel), 0, 255)

        # Convert the sharpened array back to an image
        sharpened_image = Image.fromarray(sharpened_array.astype('uint8'))

        # Destroy the "Please wait" message box
        self.end_wait()

        # Display the original and sharpened images side by side
        self.display_side_by_side(self.original_image, sharpened_image)


    def edge_detection(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please open an image first")
            return

        # Create a "Please wait" message box
        self.wait()

        # Convert image to numpy array
        image_array = np.array(self.original_image)
        height, width, channels = image_array.shape

        # Create a new empty array for the edge-detected image
        edged_array = np.zeros_like(image_array)

        # Define the laplacian kernel
        kernel = np.array([[0, 0, -1, 0, 0],
                           [0, -1, -2, -1, 0],
                           [-1, -2, 16, -2, -1],
                           [0, -1, -2, -1, 0],
                           [0, 0, -1, 0, 0]])
        kernel_size = 5
        offset = kernel_size // 2

        # Iterate over each pixel in the image
        for y in range(offset, height - offset):
            for x in range(offset, width - offset):
                for c in range(channels):
                    # Apply the laplacian kernel
                    region = image_array[y - offset:y + offset + 1, x - offset:x + offset + 1, c]
                    edged_array[y, x, c] = np.clip(np.sum(region * kernel), 0, 255)

        # Convert the edged array back to an image
        edged_array = Image.fromarray(edged_array.astype('uint8'))

        # Destroy the "Please wait" message box
        self.end_wait()

        # Display the original and edge-detected images side by side
        self.display_side_by_side(self.original_image, edged_array)


    def save_image(self):
        if self.edited_image is None:
            messagebox.showerror("Error", "Please open an image first")
            return

        # Ask user for file path
        file_path = filedialog.asksaveasfilename(defaultextension=".bmp", filetypes=[("BMP Files", "*.bmp")])
        if file_path:
            try:
                self.edited_image.save(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save the file: {str(e)}")


    # Helper functions:
    def display_side_by_side(self, image1, image2):
        # Save the edited image
        self.edited_image = image2

        # Create a new image that can hold both images side by side
        total_width = image1.width + image2.width
        max_height = max(image1.height, image2.height)
        combined = Image.new('RGB', (total_width, max_height))

        # Paste both images
        combined.paste(image1, (0, 0))
        combined.paste(image2, (image1.width, 0))

        self.display_image(combined)


    def display_image(self, image):
        self.edited_image = image

        # Clear previous image
        self.canvas.delete("all")

        # Create a PhotoImage object
        self.photo = ImageTk.PhotoImage(image)

        # Display the image on the canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        # Adjust window size to fit the image
        self.master.geometry(f"{image.width + 5}x{image.height + 5}")


    def wait(self):
        # Create a 'Please Wait' message box
        self.wait_window = Toplevel(self.master)
        self.wait_window.overrideredirect(True)
        self.wait_window.attributes('-topmost', True)

        # Calculate position to center the window
        window_width = 200
        window_height = 50
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.wait_window.geometry(f'{window_width}x{window_height}+{x}+{y}')
        self.wait_window.configure(bg='white')

        # Add message label
        label = tk.Label(
            self.wait_window,
            text="Please wait...",
            font=('Arial', 12),
            bg='white',
            pady=10
        )
        label.pack(expand=True, fill='both')

        self.wait_window.update()


    def end_wait(self):
        # Destroy the 'Please Wait' message box
        if hasattr(self, 'wait_window'):
            self.wait_window.destroy()


def main():
    root = tk.Tk()
    MiniPhotoshop(root)
    root.mainloop()


if __name__ == "__main__":
    main()
