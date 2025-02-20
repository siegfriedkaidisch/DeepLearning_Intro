# **Deep Learning Intro**

A **theoretical introduction** and **hands-on tutorial** to Deep Learning using **PyTorch Lightning**.

This repository contains:\
✅ A **presentation** covering the fundamental concepts of Deep Learning to help understand the theory behind the code.\
✅ A **practical example** based on the **MNIST dataset** that shows a typical setup for training a neural network using **PyTorch Lightning**.\
✅ A **monitoring setup** using **TensorBoard** to track training progress.

---

## **1. Setup Instructions**

### **Step 1: Download and Prepare the Dataset**

We will use the **MNIST as JPG** dataset from Kaggle. Follow these steps:

1. **Download the dataset** from Kaggle:\
   [MNIST as JPG - Kaggle](https://www.kaggle.com/datasets/scolianni/mnistasjpg)
2. **Extract the dataset** into your working directory. After extraction, your directory structure should look like this (you may need to move the directory containing images and rename it):
   ```
   ├── images/
   │   ├── 0/  (Images of digit 0)
   │   ├── 1/  (Images of digit 1)
   │   ├── 2/  (Images of digit 2)
   │   ├── ...  
   │   ├── 9/  (Images of digit 9)
   ```

---

### **Step 2: Create and Activate a Virtual Environment**

It is recommended to use a virtual environment for dependency management.

#### **Linux & macOS**

```bash
python -m venv deep_learning_demo
source deep_learning_demo/bin/activate
```

#### **Windows (PowerShell)**

```powershell
python -m venv deep_learning_demo
deep_learning_demo\Scripts\activate
```

---

### **Step 3: Install Dependencies**

#### **Install PyTorch Lightning and Torchvision**

```bash
pip install lightning torchvision
```

#### **Install TensorBoard for Logging**

```bash
pip install tensorboard
```

---

## **2. Running the Project**

### **Start TensorBoard (for Monitoring Training Progress)**

```bash
tensorboard --logdir=logs --port=6007
```

Once TensorBoard is running, open your browser and go to:\
[http://127.0.0.1:6007/](http://127.0.0.1:6007/)

---

### **Train the Model**

Execute the training script:

```bash
python mnist.py
```

This will:\
✅ Load the dataset\
✅ Train a neural network using PyTorch Lightning\
✅ Log training progress to TensorBoard\
✅ Save the model

---

## **3. Monitoring Training Progress**

- View **training loss, validation loss, etc.** in TensorBoard
- Track logs in the `logs/` directory

---

## **4. Play with it**

🔹 Experiment with different network architectures\
🔹 Try different optimizers (SGD, Adam, etc.)\
🔹 Try different activation functions (ReLU, Sigmoid)
🔹 ...

---
