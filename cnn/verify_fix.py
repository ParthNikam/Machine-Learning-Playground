import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

try:
    print("Loading data...")
    if os.path.exists('data/dataset.npz'):
        data = np.load('data/dataset.npz')
        X = data['features']
        y = data['labels']
        print(f"Loaded X shape: {X.shape}")
    else:
        print("Data file not found, creating mock data...")
        X = np.random.rand(100, 128, 128, 3) # Assuming 128 based on earlier notebook snippet
        y = np.random.randint(0, 2, 100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Initializing generators...")
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    # The fix: using .flow() properly
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=32
    )

    val_generator = val_datagen.flow(
        X_test, y_test,
        batch_size=32
    )
    
    print(f"Train generator n: {train_generator.n}")
    print(f"Val generator n: {val_generator.n}")
    
    print("Verification Successful!")

except Exception as e:
    print(f"Verification Failed: {e}")
