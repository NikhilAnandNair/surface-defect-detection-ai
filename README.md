# 🧠 Heat Sink Surface Defect Detection

This project implements a **Convolutional Neural Network (CNN)** to automatically detect surface defects on heat sinks using image data.  
It uses the **Heat Sink Surface Defect Dataset** from Kaggle, processes the data into *Defective* and *Normal* categories, trains a deep learning model, and visualizes predictions and performance metrics.

---

## 📦 Dataset

Dataset: [Heat Sink Surface Defect Dataset – Kaggle](https://www.kaggle.com/datasets/kaifengyang/heat-sink-surface-defect-dataset)

The original dataset contains:
- **`images/`** – Raw `.bmp` images of heat sinks  
- **`labels/`** – Pixel masks indicating defect regions  
- **`jsons/`** – Annotation files  
- **`read_me.txt`** – Dataset documentation  

During preprocessing, each image–mask pair is analyzed to determine whether it contains a defect.  
Images are then categorized into:


---

## ⚙️ Project Workflow

1. **Download and Extract Dataset**
   - Automatically downloaded via the Kaggle API in Colab.

2. **Preprocessing**
   - Label masks are analyzed.
   - Images containing any non-zero (defective) pixels are labeled *Defective*.
   - Clean images are labeled *Normal*.
   - Files are reorganized into a binary classification dataset.

3. **Model Architecture**
   - Built with TensorFlow / Keras Sequential API.
   - Layers:
     - 3× Convolution + MaxPooling layers  
     - Flatten → Dense(128, relu) → Dense(1, sigmoid)
   - Optimizer: `Adam`
   - Loss: `Binary Crossentropy`
   - Metrics: `Accuracy`

4. **Training**
   - Uses `ImageDataGenerator` for data augmentation and rescaling.
   - `EarlyStopping` and `ModelCheckpoint` callbacks.
   - Trains for ~10 epochs with 80/20 train–validation split.

5. **Evaluation**
   - Accuracy plot (`training_accuracy.png`)
   - Confusion matrix and classification report
   - Random prediction visualization

6. **Output**
   - Trained model: `final_model.keras`
   - Accuracy plot: `images/training_accuracy.png`

---

## 🧩 How to Run (in Google Colab)

1. Upload your `kaggle.json` file.
2. Run the setup cells to install dependencies and download the dataset.
3. Run the full notebook sequentially.
4. Once trained, download your model:

5. View training results in `images/`.

---

## 🖼️ Results

Example outputs include:
- Model Accuracy over epochs  
- Confusion Matrix  
- Random sample prediction with predicted label (*Defective* / *Normal*)

---

## 🧰 Dependencies

| Library | Version |
|----------|----------|
| Python | 3.10+ |
| TensorFlow | 2.15+ |
| NumPy | Latest |
| Matplotlib | Latest |
| Seaborn | Latest |
| scikit-learn | Latest |
| Pillow | Latest |
| tqdm | Latest |

Install them (if running locally):
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pillow tqdm kaggle
