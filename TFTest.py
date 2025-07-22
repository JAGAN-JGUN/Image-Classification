import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import load_model
from keras.utils import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix

model = load_model('bestmodel.keras')

test_gen = image_dataset_from_directory(
    "Data/test",
    labels="inferred",
    label_mode="categorical",
    image_size=(150, 150),
    batch_size=32,
    shuffle=False
)

loss, acc = model.evaluate(test_gen)
print(f"\nTest Accuracy: {acc:.4f} | Test Loss: {loss:.4f}")

y_true = np.concatenate([y for _, y in test_gen], axis=0)
y_pred_prob = model.predict(test_gen)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true_labels = np.argmax(y_true, axis=1)

class_names = test_gen.class_names

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred, target_names=class_names))

cm = confusion_matrix(y_true_labels, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
