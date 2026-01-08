from src.data_loader import load_data
from src.model import build_model

def train_asl_model():
    train_dir = "dataset/asl_alphabet_train"
    img_size = (48, 48)
    batch_size = 32
    num_classes = 29

    train_data, val_data = load_data(train_dir, img_size, batch_size)

    model = build_model((48, 48, 3), num_classes)

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=10
    )

    model.save("saved_model/asl_model.h5")
    print("Model training completed and saved!")
