# Potential error warnings (if there's a problem, it might be something related to one of these):
#     WARNING: If you get an error with the model not predicting correctly most of the time, make sure that the class names are either all lower-case or all upper-case for the first letter of each class name

import os
import tkinter as tk
from tkinter import ttk
#import sv_ttk # Use this for windows 11 theme
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from model import SimpleCNN
from train import train
from threading import Thread
import sys
import torch
from colour import Color


# Used to let print messages go to tkinter window
class OutputRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)  # Scroll to the end of the text
        self.text_widget.config(state=tk.DISABLED)

    def flush(self):
        pass

class ImageSorterApp:
    def __init__(self, root):
        self.console_stdout = sys.stdout

        self.ml_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.root = root
        self.root.resizable(False, False)
        self.root.title("Image Sorter")
        if os.path.exists("icon.ico"):
            self.root.iconbitmap("icon.ico")

        # Prompt to select the image directory
        self.dataset_directory = filedialog.askdirectory(title="Select Image Directory")
        if not self.dataset_directory:
            # User canceled, so exit the application
            root.destroy()
            return
        
        self.image_directory = self.dataset_directory + '/unsorted'

        # Variables
        try:
            self.image_classes = self.load_image_classes_from_config()  # Implement this function
            self.image_paths = self.load_image_paths()  # Implement this function
        except:
            messagebox.showerror("Error", "Dataset folder not formatted properly. See program documentation. Make sure you chose the correct dataset directory.")
            root.destroy()
            return

        # UI components
        self.image_label = ttk.Label(root, borderwidth=2, relief="solid")
        self.image_classes_frame = ttk.Frame(root, borderwidth=2, relief='flat')
        self.undo_button = ttk.Button(self.image_classes_frame, text="Undo", command=self.undo)
        self.info_packer = ttk.Frame(root, borderwidth=2, relief='flat')
        self.info_bar = ttk.Label(self.info_packer, borderwidth=2, relief='flat')
        self.ai_process_frame = ttk.Frame(root, borderwidth=2, relief="flat")
        self.train_button = ttk.Button(self.ai_process_frame, text="Train", command=self.train_model)
        self.predicted_class = ttk.Label(self.info_packer, text="Model not yet trained.", borderwidth=2, relief='flat')
        self.sorting_label = ttk.Label(self.ai_process_frame, text="Sort by:", borderwidth=2, relief='flat')
        self.sort_by_unconfidence_button = ttk.Button(self.ai_process_frame, text="Unconfidence", command=self.sort_by_unconfidence)
        self.sort_by_alphabet_button = ttk.Button(self.ai_process_frame, text="Default", command=self.sort_by_alphabeticalness)

        # Place UI components
        self.image_label.grid(row=0, column=0, padx=10, pady=5, sticky=tk.NW)
        self.image_classes_frame.grid(row=0, column=1, padx=10, pady=5, sticky=tk.NW)
        self.info_packer.grid(row=1, column=0, padx=10, pady=5, sticky=tk.NW)
        self.info_bar.pack(side='top', anchor=tk.NW, padx=10, pady=5)
        self.predicted_class.pack(side='top', anchor=tk.NW, padx=10, pady=5)

        self.ai_process_frame.grid(row=1, column=1, padx=10, pady=5, sticky=tk.NW)
        self.train_button.grid(row=0, column=0, padx=10, pady=5, sticky=tk.NW)
        self.sorting_label.grid(row=1, column=0, padx=10, pady=5, sticky=tk.NW)
        self.sort_by_unconfidence_button.grid(row=2, column=0, padx=10, pady=5, sticky=tk.NW)
        self.sort_by_alphabet_button.grid(row=3, column=0, padx=10, pady=5, sticky=tk.NW)

        self.class_buttons = {}
        # Create class buttons
        self.cmap1 = list(Color("red").range_to(Color("green"),101))

        for i, class_name in enumerate(self.image_classes):
            button = tk.Button(self.image_classes_frame, text=str(i+1)+". "+class_name, command=lambda c=class_name: self.class_button_clicked(c))
            self.class_buttons[class_name] = button
            #button.grid(row=i, column=2, padx=10, pady=5)
            button.pack(padx=10, pady=5, side='top')
            if i < 9:
                self.root.bind_all(str(i+1), lambda event, class_name=class_name: self.class_button_clicked(class_name))
        
        self.undo_button.pack(padx=10, pady=5, side='top')

        # Keybindings
        self.root.bind_all("<BackSpace>", self.undo)

        # Push to this stack when action (placing image in directory) is taken
        # Pop when reverting action
        # Format of element: ('where_moved_from', 'where_moved_to')
        self.undo_stack = []

        # Initialize model
        self.model = SimpleCNN(num_classes=len(self.image_classes), class_names=self.image_classes)
        self.model.to(self.ml_device)

        try:
            self.model.load_state_dict(torch.load(self.dataset_directory+"/trained_model_weights.pth"))
            self.model.is_trained = True
        except FileNotFoundError:
            pass

        # Display the first image
        self.next_image()

        messagebox.showinfo('ImageSorter', 'This dialog box shows because there\'s an odd bug where keyboard shortcuts in tkinter don\'t work unless I show a dialog box. Have fun sorting images!')

    def display_image(self):
        image_path = self.current_image
        image = Image.open(image_path).convert("RGB")
        photo = ImageTk.PhotoImage(image.resize((500, 500)))

        self.image_label.config(image=photo)
        self.image_label.image = photo

        if self.model.is_trained:
            prediction, confidence, probabilities = self.model.predict_class(image, self.ml_device)
            self.predicted_class.config(text=f"Predicted Class: {prediction}, Confidence: {confidence:.0%}")
            for class_name, button in self.class_buttons.items():
                rounded_confidence = max(0, min(round(100*probabilities[class_name]), 100))
                button.config(bg=self.cmap1[rounded_confidence])
    
    def next_image(self):
        if not len(self.image_paths):
            if not len(self.undo_stack):
                messagebox.showerror("No images", "There are no unsorted images in this directory.")
                root.destroy()
                return
            self.current_image = None
            undo_question_mark = messagebox.askquestion("No images left", "There are no more images. Would you like to undo?")
            if undo_question_mark == 'yes':
                self.undo()
            else:
                root.destroy()
            return
        self.current_image = self.image_paths.pop()
        self.info_bar.config(text=f'{len(self.image_paths)} images left. {os.path.basename(self.current_image)}')
        self.display_image()

    def undo(self, event=None):
        # Implement undo functionality

        # Nothing to undo
        if not len(self.undo_stack):
            messagebox.showerror("Error", "Nothing to undo.")
            return
        
        # Move image back to where it came from and display it
        source_path, destination_path = self.undo_stack.pop()
        os.rename(destination_path, source_path)
        if not self.current_image is None:
            self.image_paths.append(self.current_image)
        self.image_paths.append(source_path)
        self.next_image()

    def class_button_clicked(self, class_name):
        # Implement class button functionality
        source_path = self.current_image
        destination_path = self.dataset_directory + f'/sorted/{class_name}/' + os.path.basename(source_path)
        self.undo_stack.append((source_path, destination_path))
        os.rename(source_path, destination_path)

        self.next_image()

    def load_image_classes_from_config(self):
        # Implement loading image classes from config file
        # Return a list of class names
        config_file = self.dataset_directory + '/classes.txt'
        with open(config_file) as f:
            image_classes = f.read().splitlines()
        
        for class_name in image_classes:
            os.makedirs(self.dataset_directory + f'/sorted/{class_name}', exist_ok=True)
        return image_classes

    def load_image_paths(self):
        # Implement loading image paths from the selected directory
        # Return a list of image file paths
        if not os.path.exists(self.image_directory):
            return []

        image_files = [f for f in os.listdir(self.image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        image_paths = [os.path.join(self.image_directory, f) for f in image_files]
        image_paths.reverse() # So I can .pop() from it
        return image_paths
    
    def train_model(self):
        for class_name in self.image_classes:
            if not os.listdir(self.dataset_directory + f'/sorted/{class_name}'):
                messagebox.showerror("Error", f"You must have at least one image in each category to train the model. You currently have none in {class_name}, possibly other classes are empty as well.")
                return
        #self.undo_stack = []
        self.model.is_trained = True
        train_proc = lambda: train(self.model, self.dataset_directory+'/sorted/', num_epochs=5, lr=0.001, save_path=self.dataset_directory+'/trained_model_weights.pth')
        self.run_with_loading_window(train_proc)
    
    def run_with_loading_window(self, proc):
        loading_dialog = tk.Toplevel(root)
        loading_dialog.title("Loading...")
        loading_dialog.grab_set()

        # Create a Text widget to display the output
        output_text = tk.Text(loading_dialog, wrap=tk.WORD, height=10, width=40, bg="black", fg="white", state=tk.DISABLED)
        output_text.pack(padx=10, pady=10)

        # Redirect stdout to the Text widget
        sys.stdout = OutputRedirector(output_text)

        # Run the function in a separate thread
        thread = Thread(target=proc)
        thread.start()

        # Check the status of the thread periodically
        self.check_thread_status(loading_dialog, thread)

    def check_thread_status(self, loading_dialog, thread):
        if thread.is_alive():
            # Continue checking the thread status after 100 milliseconds
            root.after(100, lambda: self.check_thread_status(loading_dialog, thread))
        else:
            # Close the loading dialog when the thread is finished
            loading_dialog.grab_release()
            loading_dialog.destroy()
            sys.stdout = self.console_stdout
    
    def _sort_by_uncondidence_key(self, image_path):
        image = Image.open(image_path).convert("RGB")
        _, confidence, _ = self.model.predict_class(image, self.ml_device)
        return confidence

    def _sort_by_unconfidence(self):
        print("Sorting images by unconfidence...")
        self.image_paths.sort(key=lambda x: self._sort_by_uncondidence_key(x), reverse=True)
        print("Done.")

    def sort_by_unconfidence(self):
        if not self.model.is_trained:
            messagebox.showerror("Error", "Model must be trained prior to sorting by unconfidence.")
            return
        self.run_with_loading_window(self._sort_by_unconfidence)
    
    def sort_by_alphabeticalness(self):
        self.image_paths.sort(reverse=True)
    



if __name__ == "__main__":
    root = tk.Tk()
    #sv_ttk.set_theme("dark")
    style = ttk.Style()
    style.theme_use('vista')
    app = ImageSorterApp(root)
    root.mainloop()
