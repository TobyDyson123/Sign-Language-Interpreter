import cv2
import mediapipe as mp
from tkinter import *
from PIL import Image, ImageTk
from globals import *

# Function to toggle the drawing of landmarks
def toggle_landmarks():
    global draw_landmarks
    draw_landmarks = not draw_landmarks

# Initialize the flag for drawing landmarks
draw_landmarks = True

# Function to update the image in the Tkinter window
def update_image():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    image, results = mediapipe_detection(frame, holistic)
    if draw_landmarks:
        draw_styled_landmarks(image, results)
    cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, update_image)

# Initialize Tkinter window
root = Tk()
root.title("Sign Language Translator")

# Create the heading
heading = Label(root, text="Sign Language Translator", font=('Arial', 24))
heading.pack()

# Create a frame for the camera feed and translated text
main_frame = Frame(root)
main_frame.pack()

# Create and place the video feed label
lmain = Label(main_frame)
lmain.pack(side=LEFT)

# Create a frame for the translated text and checkbox
text_frame = Frame(main_frame)
text_frame.pack(side=RIGHT, fill=Y, expand=True)

# Create and place the translated text label and text box
translated_text_label = Label(text_frame, text="Translated Text:")
translated_text_label.pack()
translated_text = Text(text_frame, height=5, width=50)
translated_text.pack()

# Create and place the checkbox for landmarks
landmarks_var = BooleanVar(value=True)
landmarks_checkbox = Checkbutton(text_frame, text="Enable Landmarks?", var=landmarks_var, command=toggle_landmarks)
landmarks_checkbox.pack(anchor=W)

# Initialize OpenCV capture
cap = cv2.VideoCapture(0)

# Set desired width and height for the camera feed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Actual resolution: {actual_width}x{actual_height}")

# Set mediapipe model
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start the update process
update_image()

# Start the Tkinter main loop
root.mainloop()

# Release resources when the window is closed
cap.release()
cv2.destroyAllWindows()
