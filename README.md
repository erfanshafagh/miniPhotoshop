# miniPhotoshop
Mini Photoshop is a Python-based image editing application with a user-friendly GUI built using Tkinter. It provides essential image processing features such as opening BMP files, grayscale conversion, ordered dithering, auto-level adjustment, blur, and more. The application uses PIL, NumPy, and Struct for efficient image processing.

## Features

### Core Features
- **Open BMP File**: Load BMP images (24-bit, max size 704x576).
- **Grayscale**: Convert the image to grayscale using the luminosity method.
- **Ordered Dithering**: Apply ordered dithering using a 4x4 Bayer matrix.
- **Auto Level**: Enhance image contrast automatically.
- **Exit**: Close the application.

### Optional Features
- **Blur**: Smooth the image using a 6x6 averaging kernel.
- **Sharpening**: Enhance image details with a 3x3 sharpening kernel.
- **Edge Detection**: Highlight edges using a 5x5 Laplacian kernel.
- **Save Image**: Save the modified image.

## Libraries Used
- **Tkinter**: For building the graphical user interface.
- **PIL (Pillow)**: For image processing tasks.
- **NumPy**: For efficient array and matrix operations.
- **Struct**: For reading BMP file headers.

## How to Run

To get started with Mini Photoshop, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/erfanshafagh/miniPhotoshop.git
    ```

2. Run the program :
    ```bash
    python3 miniphotoshop.py
    ```

## Contributing

If you find any issues or have suggestions for improvement, feel free to open an [issue](https://github.com/erfanshafagh/miniPhotoshop/issues) or create a [pull request](https://github.com/erfanshafagh/miniPhotoshop/pulls).
