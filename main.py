import cv2
import numpy as np
import mediapipe as mp
from tkinter import *
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from globals import *

model = load_model('action.h5')

# Initialize MediaPipe holistic model
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

colors = [(245,117,16), (117,245,16), (16,117,245)]  # Update with desired colors

# Global variable to control the display of probability visualization
display_prob_viz = True

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    if display_prob_viz:
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num % len(colors)], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

# Function to toggle the probability visualization
def toggle_prob_viz():
    global display_prob_viz
    display_prob_viz = not display_prob_viz

# Function to update the image in the Tkinter window and make predictions
def update_image():
    global sequence, sentence, predictions, draw_landmarks
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_image)
        return

    frame = cv2.flip(frame, 1)
    image, results = mediapipe_detection(frame, holistic)
    if draw_landmarks:
        draw_styled_landmarks(image, results)

    # Prediction logic
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]

    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        predictions.append(np.argmax(res))
        # Viz probabilities
        image = prob_viz(res, actions, image, colors)
        
        if np.unique(predictions[-10:])[0] == np.argmax(res) and res[np.argmax(res)] > threshold:
            if len(sentence) > 0 and actions[np.argmax(res)] != sentence[-1]:
                sentence.append(actions[np.argmax(res)])
            elif len(sentence) == 0:
                sentence.append(actions[np.argmax(res)])

    if len(sentence) > 5:
        sentence = sentence[-5:]

    # Update the GUI with the predicted action
    translated_text.delete("1.0", END)
    translated_text.insert(END, ' '.join(sentence))

    cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, update_image)

# Toggle landmarks drawing
def toggle_landmarks():
    global draw_landmarks
    draw_landmarks = not draw_landmarks

# Initialize global variables for prediction
sequence = []
sentence = []
predictions = []
threshold = 0.5
draw_landmarks = True

# Initialize the Tkinter GUI
root = Tk()
root.title("Sign Language Translator")

# Create a frame for the camera feed and translated text
main_frame = Frame(root)
main_frame.pack(pady=15)

# Create and place the video feed label
lmain = Label(main_frame)
lmain.grid(row=0, column=0, padx=10)

# Create a text widget in the GUI for displaying the predicted actions
translated_text = Text(main_frame, height=10, width=50)
translated_text.grid(row=0, column=1, padx=10)

# Create and place the checkbox for landmarks
landmarks_var = BooleanVar(value=True)
landmarks_checkbox = Checkbutton(main_frame, text="Enable Landmarks?", var=landmarks_var, command=toggle_landmarks)
landmarks_checkbox.grid(row=1, column=1, sticky='w', padx=10)

prob_viz_var = BooleanVar(value=True)
prob_viz_checkbox = Checkbutton(main_frame, text="Show Probabilities?", var=prob_viz_var, command=toggle_prob_viz)
prob_viz_checkbox.grid(row=2, column=1, sticky='w', padx=10)

# Initialize and configure the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Start the GUI update process
update_image()

# Start the Tkinter main loop
root.mainloop()

# Release resources when the window is closed
cap.release()
cv2.destroyAllWindows()