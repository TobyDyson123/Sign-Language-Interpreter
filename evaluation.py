#####################################################################
#
# Preprocess Data and Create Labels and Features
#
#####################################################################

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from globals import *

label_map = {label:num for num, label in enumerate(actions)}
# Example output: {'hello': 0, 'thanks': 1, 'iloveyou': 2}

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

#####################################################################
#
# Evaluation using Confusion Matrix and Accuracy
#
#####################################################################

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model

model = load_model('action.h5')

yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

# Confusion Matrix
cfmatrix = multilabel_confusion_matrix(ytrue, yhat)
print (cfmatrix)

# Accuracy
accuracy = accuracy_score(ytrue, yhat)
print(f"Accuracy: {accuracy:.2f}")

# Precision, Recall, and F1 Score
precision = precision_score(ytrue, yhat, average='weighted')
recall = recall_score(ytrue, yhat, average='weighted')
f1 = f1_score(ytrue, yhat, average='weighted')
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")