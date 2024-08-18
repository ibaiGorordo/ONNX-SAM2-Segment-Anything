import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

from sam2 import SAM2Image, draw_masks, colors
from imread_from_url import imread_from_url

class ImageAnnotationApp:
    def __init__(self, root, sam2: SAM2Image):
        self.root = root
        self.sam2 = sam2
        self.root.title("Image Annotation App")
        self.root.geometry("1500x900")

        # Image and canvas initialization
        self.image = None
        self.tk_image = None
        self.canvas = tk.Canvas(root, width=1280, height=720)
        self.canvas.pack(side=tk.LEFT)

        # Browse button
        self.browse_button = tk.Button(root, text="Browse Image", command=self.browse_image)
        self.browse_button.pack()

        # Sidebar for labels
        self.label_frame = tk.Frame(root)
        self.label_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.label_listbox = tk.Listbox(self.label_frame)
        self.label_listbox.pack()

        self.add_label_button = tk.Button(self.label_frame, text="Add Label", command=self.add_label)
        self.add_label_button.pack()

        self.remove_label_button = tk.Button(self.label_frame, text="Remove Label", command=self.remove_label)
        self.remove_label_button.pack()

        self.selected_label = None
        self.points = []
        self.label_ids = []
        self.label_colors = {}

        self.canvas.bind("<Button-1>", self.on_positive_point)
        self.canvas.bind("<Button-3>", self.on_negative_point)

        self.add_label(0)  # Add default label 1
        img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e2/Dexter_professionellt_fotograferad.jpg/1280px-Dexter_professionellt_fotograferad.jpg"
        self.image = imread_from_url(img_url)
        self.mask_image = self.image.copy()
        self.sam2.set_image(self.image)
        self.display_image()

    def browse_image(self):
        if self.image is not None:
            self.reset()

        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            self.image = cv2.imread(self.image_path)
            self.mask_image = self.image.copy()
            self.sam2.set_image(self.image)
            self.display_image()

    def display_image(self):
        if self.mask_image.shape[0] == 0:
            return

        # Convert the image to RGB (from BGR)
        rgb_image = cv2.cvtColor(self.mask_image, cv2.COLOR_BGR2RGB)
        # Convert the image to PIL format
        pil_image = Image.fromarray(rgb_image)
        self.tk_image = ImageTk.PhotoImage(image=pil_image)
        self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.draw_points()

    def add_label(self, label_id: int = None):
        if label_id is None:
            max_label = max(self.label_ids) if self.label_ids else 0

            # If the number of labels is less than the maximum label, use the next available label, otherwise use the next number
            if len(self.label_ids) == 0:
                label_id = 0
            elif len(self.label_ids) <= max_label:
                label_id = next(i for i in range(0, max_label + 1) if i not in self.label_ids)

            else:
                label_id = max_label + 1

        label = f"Label {label_id}"

        self.label_listbox.insert(tk.END, label)
        self.label_listbox.bind("<<ListboxSelect>>", self.select_label)
        self.label_listbox.selection_clear(0, tk.END)
        self.label_listbox.selection_set(tk.END)
        self.selected_label = label
        self.label_ids.append(label_id)

        b, g, r = colors[label_id]
        color = f'#{int(r):02x}{int(g):02x}{int(b):02x}'
        self.label_colors[label] = color

        # Set the background color of the listbox item
        self.label_listbox.itemconfig(tk.END, {'bg': color, 'fg': 'white'})

        # Sort the labels in the Listbox
        self.sort_labels()

    def select_label(self, event):
        widget = event.widget
        selection = widget.curselection()
        if selection:
            self.selected_label = widget.get(selection[0])

    def remove_label(self):
        selection = self.label_listbox.curselection()
        if selection:
            label = self.label_listbox.get(selection[0])
            self.label_listbox.delete(selection[0])

            label_id = int(label.split()[-1])
            self.label_ids.remove(label_id)

            # Remove points associated with this label
            points_to_remove = [point for point in self.points if point[3] == label]
            for point in points_to_remove:
                self.sam2.remove_point((point[1], point[2]), label_id)
                self.canvas.delete(point[0])
                self.points.remove(point)

            masks = self.sam2.get_masks()
            self.mask_image = draw_masks(self.image, masks)
            self.display_image()

            self.selected_label = None
            if self.label_listbox.size() > 0:
                self.label_listbox.selection_set(0)
                self.selected_label = self.label_listbox.get(0)

            # Sort the labels in the Listbox
            self.sort_labels()

    def sort_labels(self):
        labels = list(self.label_listbox.get(0, tk.END))
        labels.sort(key=lambda x: int(x.split()[-1]))
        self.label_listbox.delete(0, tk.END)
        for label in labels:
            self.label_listbox.insert(tk.END, label)
            # Reapply the background color
            self.label_listbox.itemconfig(tk.END, {'bg': self.label_colors[label], 'fg': 'white'})

    def on_positive_point(self, event):
        if self.image is None:
            return

        # Check if the point is close to an existing point for deletion
        x, y = event.x, event.y
        closest_point = None
        closest_distance = float('inf')

        for point in self.points:
            _, px, py, _, _ = point
            distance = (x - px) ** 2 + (y - py) ** 2
            if distance < closest_distance:
                closest_distance = distance
                closest_point = point

        if closest_point and closest_distance < 10 ** 2:  # Within 10 pixels
            self.canvas.delete(closest_point[0])
            self.points.remove(closest_point)
            label_id = int(closest_point[3].split()[-1])

            self.sam2.remove_point((closest_point[1], closest_point[2]), label_id)
            print(f"Removed point at ({closest_point[1]}, {closest_point[2]})")

        elif self.selected_label:
            x, y = event.x, event.y
            label_id = int(self.selected_label.split()[-1])

            color = f'#{0:02x}{255:02x}{0:02x}'

            radius = 4
            point = self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline=color)
            self.points.append((point, x, y, self.selected_label, True))

            self.sam2.add_point((x, y), True, label_id)
            print(f"Added point at ({x}, {y}) with label '{self.selected_label}'")

        masks = self.sam2.get_masks()
        self.mask_image = draw_masks(self.image, masks)
        self.display_image()

    def on_negative_point(self, event):
        if self.image is None or not self.selected_label:
            return

        print(f"Right click at ({event.x}, {event.y})")
        x, y = event.x, event.y
        label_id = int(self.selected_label.split()[-1])

        b, g, r = colors[label_id]
        color = f'#{255:02x}{0:02x}{0:02x}'

        radius = 3
        point = self.canvas.create_rectangle(x - radius*3, y - radius, x + radius*3, y + radius, fill=color, outline=color)
        self.points.append((point, x, y, self.selected_label, False))

        self.sam2.add_point((x, y), False, label_id)
        masks = self.sam2.get_masks()
        self.mask_image = draw_masks(self.image, masks)
        self.display_image()

        print(f"Added point at ({x}, {y}) with label '{self.selected_label}'")


    def draw_points(self):
        for point in self.points:
            _, x, y, label, is_valid = point

            radius = 4
            if is_valid:
                color = f'#{0:02x}{255:02x}{0:02x}'
            else:
                color = f'#{255:02x}{0:02x}{0:02x}'
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline=color)


    def generate_color(self):
        import random
        r = lambda: random.randint(0, 255)
        return f'#{r():02x}{r():02x}{r():02x}'

    def reset(self):
        self.image = None
        self.mask_image = None
        self.tk_image = None
        self.canvas.delete("all")
        self.points = []
        self.label_listbox.delete(0, tk.END)
        self.label_ids = []
        self.selected_label = None
        self.add_label(0)


if __name__ == "__main__":
    root = tk.Tk()

    encoder_model_path = "models/sam2_hiera_base_plus_encoder.onnx"
    decoder_model_path = "models/sam2_hiera_base_plus_decoder.onnx"
    sam2 = SAM2Image(encoder_model_path, decoder_model_path)

    app = ImageAnnotationApp(root, sam2)
    root.mainloop()