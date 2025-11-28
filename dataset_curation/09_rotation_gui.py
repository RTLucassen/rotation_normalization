import math
import json
import os

import tkinter as tk
from PIL import Image, ImageTk
from PIL import ImageDraw
from tkinter import filedialog


class ImageRotator:
    """
    A GUI-based tool for loading, displaying, resizing, and interactively rotating images.
    Supports selecting and navigating through multiple images.
    Saves rotation states in a JSON file.
    Allows skipping images by setting their rotation to None.
    """

    ROTATION_FILE = "image_rotations_HE.json"  # File to save/load rotation states

    def __init__(self, root):
        self.root = root
        self.root.title("Image Rotation Tool")
        self.center_window(800, 600)

        # Canvas for image display
        self.canvas = tk.Canvas(root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Button to load images
        self.load_button = tk.Button(root, text="Load Images", command=self.load_images)
        self.load_button.pack()

        # Label to show current rotation
        self.angle_label = tk.Label(root, text="Rotation: 0°")
        self.angle_label.pack()

        self.filename_label = tk.Label(root, text="No image loaded")
        self.filename_label.pack()

        # Navigation buttons: previous, skip, next
        self.nav_frame = tk.Frame(root)
        self.nav_frame.pack(pady=(0,5))

        self.prev_button = tk.Button(self.nav_frame, text="<< Previous", command=self.show_previous_image, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.skip_quality_button = tk.Button(self.nav_frame, text="Low quality cross-section", command=lambda: self.skip_image("low_quality"), state=tk.DISABLED)
        self.skip_quality_button.pack(side=tk.LEFT, padx=5)

        self.skip_rotation_button = tk.Button(self.nav_frame, text="No suitable rotation", command=lambda: self.skip_image("no_rotation"), state=tk.DISABLED)
        self.skip_rotation_button.pack(side=tk.LEFT, padx=5)

        self.skip_control_button = tk.Button(self.nav_frame, text="Control", command=lambda: self.skip_image("control"), state=tk.DISABLED)
        self.skip_control_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(self.nav_frame, text="Next >>", command=self.show_next_image, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=5)

        # State variables
        self.image_paths = []
        self.current_image_index = 0
        self.image = None
        self.original_image = None
        self.tk_image = None
        self.total_rotation = 0
        self.start_angle = None
        self.user_resizing = False

        # Load rotation state from file (or empty if not available)
        self.rotation_dict = self.load_rotation_dict()

        # Set window minimum size and handle close
        self.root.update_idletasks()
        self.root.minsize(400, 400)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Bind events
        self.root.bind("<Configure>", self.on_resize)
        self.root.bind("<ButtonPress-1>", self.start_user_resize)
        self.root.bind("<ButtonRelease-1>", self.stop_user_resize)
        self.canvas.bind("<ButtonPress-1>", self.start_rotate)
        self.canvas.bind("<B1-Motion>", self.do_rotate)
        # Key bindings for navigation
        self.root.bind("<Left>", lambda event: self.show_previous_image())
        self.root.bind("a", lambda event: self.show_previous_image())
        self.root.bind("<Right>", lambda event: self.show_next_image())
        self.root.bind("d", lambda event: self.show_next_image())
        self.root.bind("q", lambda event: self.skip_image("low_quality"))
        self.root.bind("r", lambda event: self.skip_image("no_rotation"))
        self.root.bind("c", lambda event: self.skip_image("control"))

    def center_window(self, width, height):
        """Center the application window on the screen."""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def load_rotation_dict(self):
        """Load previously saved rotation dictionary from JSON."""
        if os.path.exists(self.ROTATION_FILE):
            with open(self.ROTATION_FILE, "r") as f:
                dictionary = json.load(f)
                print('Previously saved rotations loaded')
                return dictionary
        return {}

    def save_rotation_dict(self):
        """Save the current rotation dictionary to JSON."""
        with open(self.ROTATION_FILE, "w") as f:
            json.dump(self.rotation_dict, f, indent=2)

    def on_close(self):
        """Handle window close: save rotation state then exit."""
        self.save_rotation_dict()
        self.root.destroy()

    def load_images(self):
        """Open file dialog to select multiple images."""
        file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if file_paths:
            self.image_paths = list(file_paths)
            self.current_image_index = 0
            self.load_image_at_index(self.current_image_index)

    def load_image_at_index(self, index):
        """Load and display image at a given index."""
        if 0 <= index < len(self.image_paths):
            image_path = self.image_paths[index]
            self.filename_label.config(text=os.path.basename(image_path))  # <--- new line
            self.original_image = Image.open(image_path)
            self.pad_image()
            self.resize_image_to_fit_canvas()
            rotation_entry = self.rotation_dict.get(image_path)
            if isinstance(rotation_entry, dict) and "skipped" in rotation_entry:
                self.total_rotation = 0
                self.angle_label.config(text=f"Skipped: {rotation_entry['skipped']}")
            else:
                self.total_rotation = rotation_entry or 0
                self.angle_label.config(text=f"Rotation: {int(self.total_rotation)}°")
            self.display_image()
            self.update_nav_buttons()

    def show_previous_image(self):
        """Go to previous image if available."""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image_at_index(self.current_image_index)

    def show_next_image(self):
        """Go to next image if available."""
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.load_image_at_index(self.current_image_index)

    def skip_image(self, reason):
        """Skip current image by assigning None and recording reason."""
        if self.image_paths:
            current_path = self.image_paths[self.current_image_index]
            self.rotation_dict[current_path] = {"skipped": reason}
            self.show_next_image()

    def update_nav_buttons(self):
        """Enable/disable navigation buttons depending on position."""
        has_images = len(self.image_paths) > 0
        can_go_prev = self.current_image_index > 0
        can_go_next = self.current_image_index < len(self.image_paths) - 1

        self.prev_button.config(state=tk.NORMAL if can_go_prev else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if can_go_next else tk.DISABLED)
        self.skip_quality_button.config(state=tk.NORMAL if has_images else tk.DISABLED)
        self.skip_rotation_button.config(state=tk.NORMAL if has_images else tk.DISABLED)
        self.skip_control_button.config(state=tk.NORMAL if has_images else tk.DISABLED)

    def pad_image(self):
        """Pad image to square so it doesn't get cropped during rotation."""
        if self.original_image:
            w, h = self.original_image.size
            diagonal = int(math.sqrt(w**2 + h**2))
            padded_image = Image.new("RGBA", (diagonal, diagonal), (0, 0, 0, 0))
            paste_x = (diagonal - w) // 2
            paste_y = (diagonal - h) // 2
            padded_image.paste(self.original_image, (paste_x, paste_y))
            self.original_image = padded_image

    def resize_image_to_fit_canvas(self):
        """Resize image to fit canvas while maintaining aspect ratio."""
        max_width = self.canvas.winfo_width()
        max_height = self.canvas.winfo_height()
        if max_width > 0 and max_height > 0:
            w, h = self.original_image.size
            scale_factor = min(max_width / w, max_height / h)
            new_size = (int(w * scale_factor), int(h * scale_factor))
            self.image = self.original_image.resize(new_size, Image.LANCZOS)

    def display_image(self, opacity=0.8, num_horizontal_lines=10, num_vertical_lines=15):
        """Rotate and draw image onto the canvas with adjustable opacity and a generated grid background."""
        if self.image:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # Generate white background with black grid lines
            bg = Image.new("RGBA", (canvas_width, canvas_height), (245, 245, 245, 245))
            draw = ImageDraw.Draw(bg)

            # Draw horizontal lines
            if num_horizontal_lines > 0:
                spacing_y = canvas_height / (num_horizontal_lines + 1)
                for i in range(1, num_horizontal_lines + 1):
                    y = int(i * spacing_y)
                    draw.line([(0, y), (canvas_width, y)], fill=(0, 0, 0), width=1)

            # Draw vertical lines
            if num_vertical_lines > 0:
                spacing_x = canvas_width / (num_vertical_lines + 1)
                for i in range(1, num_vertical_lines + 1):
                    x = int(i * spacing_x)
                    draw.line([(x, 0), (x, canvas_height)], fill=(0, 0, 0), width=1)

            self.bg_image = bg

            rotated = self.image.rotate(-self.total_rotation, resample=Image.BICUBIC,
                                        center=(self.image.width // 2, self.image.height // 2))
            if rotated.mode != "RGBA":
                rotated = rotated.convert("RGBA")

            # Apply opacity
            alpha = rotated.split()[3].point(lambda p: int(p * opacity))
            rotated.putalpha(alpha)

            # Composite rotated image onto background
            combined = self.bg_image.copy()
            paste_x = (self.bg_image.width - rotated.width) // 2
            paste_y = (self.bg_image.height - rotated.height) // 2
            combined.paste(rotated, (paste_x, paste_y), rotated)

            self.tk_image = ImageTk.PhotoImage(combined)
            self.canvas.delete("all")
            self.center_x = canvas_width // 2
            self.center_y = canvas_height // 2
            self.canvas.create_image(self.center_x, self.center_y, image=self.tk_image, anchor=tk.CENTER)

    def on_resize(self, event):
        """Re-render image on window resize."""
        if self.original_image:
            self.resize_image_to_fit_canvas()
            self.display_image()

    def start_user_resize(self, event):
        """Mark start of window resizing (not used here but may be useful)."""
        self.user_resizing = True

    def stop_user_resize(self, event):
        """Mark end of window resizing (not used here but may be useful)."""
        self.user_resizing = False

    def start_rotate(self, event):
        """Capture initial mouse angle when dragging starts."""
        if self.image:
            self.start_angle = math.atan2(event.y - self.center_y, event.x - self.center_x)

    def do_rotate(self, event):
        """Rotate image based on mouse dragging motion."""
        if self.image and self.start_angle is not None:
            current_angle = math.atan2(event.y - self.center_y, event.x - self.center_x)
            delta_angle = math.degrees(current_angle - self.start_angle)
            self.total_rotation = (self.total_rotation + delta_angle) % 360
            self.start_angle = current_angle
            self.angle_label.config(text=f"Rotation: {int(self.total_rotation)}°")
            self.display_image()

            # Save the rotation for current image
            current_path = self.image_paths[self.current_image_index]
            self.rotation_dict[current_path] = self.total_rotation

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRotator(root)
    root.mainloop()