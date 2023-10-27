import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

class ImagePreprocessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Preprocessing App")

        self.image = None
        self.filtered_image = None

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.save_button = tk.Button(root, text="Save Image", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack()

        self.filter_label = tk.Label(root, text="Select a Filter:")
        self.filter_label.pack()

        self.filter_var = tk.StringVar()
        self.filter_var.set("Original")

        self.filter_menu = tk.OptionMenu(root, self.filter_var, "Original", "Grayscale", "Blur", "Edge Detection")
        self.filter_menu.pack()

        self.apply_button = tk.Button(root, text="Apply Filter", command=self.apply_filter)
        self.apply_button.pack()

        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.update_display()
            self.save_button.config(state=tk.NORMAL)

    def save_image(self):
        if self.filtered_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg")
            if file_path:
                cv2.imwrite(file_path, self.filtered_image)
                messagebox.showinfo("Info", "Image saved successfully.")

    def apply_filter(self):
        if self.image is not None:
            selected_filter = self.filter_var.get()

            if selected_filter == "Grayscale":
                self.filtered_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            elif selected_filter == "Blur":
                self.filtered_image = cv2.GaussianBlur(self.image, (5, 5), 0)
            elif selected_filter == "Edge Detection":
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                self.filtered_image = cv2.Canny(gray, 100, 200)

            self.update_display()

    def update_display(self):
        if self.filtered_image is not None:
            image = cv2.cvtColor(self.filtered_image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB) if self.image is not None else None
    
        if image is not None:
            height, width, channels = image.shape
            image = cv2.resize(image, (400, 400))
            
            # Convert the image to PhotoImage format (using PIL)
            from PIL import Image, ImageTk
            image_pil = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image=image_pil)
            
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.photo = photo
    

if __name__ == "__main__":
    root = tk.Tk()
    app = ImagePreprocessingApp(root)
    root.mainloop()
