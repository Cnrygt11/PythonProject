# CNN Tabanlı Yüz Duygu Analizi Projesi - Google Colab
# jonathanoheix/face-expression-recognition-dataset kullanımı

# Gerekli kütüphanelerin import edilmesi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
from google.colab import drive

# Google Drive'ı mount et (gerekirse)
# drive.mount('/content/drive')

# Veri setinin yolu
base_path = '/content/sample_data/images/images'
train_path = os.path.join(base_path, 'train')
validation_path = os.path.join(base_path, 'validation')

# Duygu etiketleri (veri setindeki klasör isimleri)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
num_classes = len(emotion_labels)

print(f"Toplam duygu kategorisi: {num_classes}")
print(f"Kategoriler: {emotion_labels}")

# Veri seti hakkında bilgi alma
def get_dataset_info():
    """
    Veri seti hakkında detaylı bilgi verir
    """
    print("TRAIN VERİ SETİ DAĞILIMI:")
    print("-" * 40)
    train_counts = {}
    total_train = 0
    
    for emotion in emotion_labels:
        emotion_path = os.path.join(train_path, emotion)
        if os.path.exists(emotion_path):
            count = len(os.listdir(emotion_path))
            train_counts[emotion] = count
            total_train += count
            print(f"{emotion.capitalize()}: {count} görüntü")
    
    print(f"\nToplam eğitim görüntüsü: {total_train}")
    
    print("\nVALIDATION VERİ SETİ DAĞILIMI:")
    print("-" * 40)
    val_counts = {}
    total_val = 0
    
    for emotion in emotion_labels:
        emotion_path = os.path.join(validation_path, emotion)
        if os.path.exists(emotion_path):
            count = len(os.listdir(emotion_path))
            val_counts[emotion] = count
            total_val += count
            print(f"{emotion.capitalize()}: {count} görüntü")
    
    print(f"\nToplam validasyon görüntüsü: {total_val}")
    print(f"Genel toplam: {total_train + total_val}")
    
    return train_counts, val_counts

# CNN Model Mimarisi
def create_emotion_cnn_model(input_shape=(48, 48, 3), num_classes=7):
    """
    Duygu analizi için optimize edilmiş CNN modeli
    """
    model = Sequential([
        # İlk Konvolüsyon Bloğu
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # İkinci Konvolüsyon Bloğu
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Üçüncü Konvolüsyon Bloğu
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Dördüncü Konvolüsyon Bloğu
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.5),
        
        # Tam Bağlantılı Katmanlar
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax', name='predictions')
    ])
    
    return model

# Veri generatorları oluşturma
def create_data_generators(batch_size=32, img_size=(48, 48)):
    """
    ImageDataGenerator kullanarak veri yükleme ve artırma
    """
    # Eğitim verisi için data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validasyon verisi için sadece normalizasyon
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Train generator
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=emotion_labels,
        shuffle=True
    )
    
    # Validation generator
    validation_generator = validation_datagen.flow_from_directory(
        validation_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=emotion_labels,
        shuffle=False
    )
    
    return train_generator, validation_generator

# Model eğitimi
def train_emotion_model(epochs=50, batch_size=32):
    """
    Modeli eğitir ve sonuçları döndürür
    """
    print("Model eğitimi başlıyor...")
    print("=" * 50)
    
    # Veri generatorları oluştur
    train_gen, val_gen = create_data_generators(batch_size=batch_size)
    
    # Model oluştur
    model = create_emotion_cnn_model()
    
    # Model derleme
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Model özeti
    print("Model Mimarisi:")
    model.summary()
    
    # Callback'ler
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001,
            verbose=1
        ),
        ModelCheckpoint(
            'best_emotion_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Model eğitimi
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_gen.samples // batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history, train_gen, val_gen

# Eğitim geçmişini görselleştirme
def plot_training_history(history):
    """
    Eğitim ve validasyon metriklerini görselleştirir
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Eğitim Accuracy', color='blue')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validasyon Accuracy', color='red')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Eğitim Loss', color='blue')
    axes[0, 1].plot(history.history['val_loss'], label='Validasyon Loss', color='red')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning Rate (eğer kayıtlı ise)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'], label='Learning Rate', color='green')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    else:
        axes[1, 0].text(0.5, 0.5, 'Learning Rate verisi mevcut değil', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # Accuracy karşılaştırması (bar chart)
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    axes[1, 1].bar(['Eğitim', 'Validasyon'], [final_train_acc, final_val_acc], 
                   color=['blue', 'red'], alpha=0.7)
    axes[1, 1].set_title('Final Accuracy Karşılaştırması')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_ylim(0, 1)
    
    # Değerleri bar üzerine yaz
    axes[1, 1].text(0, final_train_acc + 0.01, f'{final_train_acc:.3f}', 
                   ha='center', va='bottom')
    axes[1, 1].text(1, final_val_acc + 0.01, f'{final_val_acc:.3f}', 
                   ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Confusion Matrix ve detaylı analiz
def evaluate_model(model, val_generator):
    """
    Model performansını detaylı analiz eder
    """
    # Tahminleri al
    val_generator.reset()
    predictions = model.predict(val_generator, steps=val_generator.samples // val_generator.batch_size + 1)
    
    # Gerçek etiketleri al
    y_true = val_generator.classes[:len(predictions)]
    y_pred = np.argmax(predictions, axis=1)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[label.capitalize() for label in emotion_labels],
                yticklabels=[label.capitalize() for label in emotion_labels])
    plt.title('Confusion Matrix - Duygu Sınıflandırması')
    plt.xlabel('Tahmin Edilen Duygu')
    plt.ylabel('Gerçek Duygu')
    plt.show()
    
    # Sınıflandırma raporu
    print("DETAYLI SINIFLANDIRMA RAPORU:")
    print("=" * 50)
    report = classification_report(y_true, y_pred, 
                                 target_names=[label.capitalize() for label in emotion_labels],
                                 digits=4)
    print(report)
    
    # Her sınıf için accuracy hesapla
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    print("\nSINIF BAZLI ACCURACY DEĞERLERI:")
    print("-" * 40)
    for i, emotion in enumerate(emotion_labels):
        print(f"{emotion.capitalize()}: {class_accuracy[i]:.4f} ({class_accuracy[i]*100:.2f}%)")
    
    overall_accuracy = np.sum(cm.diagonal()) / np.sum(cm)
    print(f"\nGenel Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    return overall_accuracy, class_accuracy

# Örnek tahminleri görselleştirme
def visualize_predictions(model, val_generator, num_images=12):
    """
    Rastgele örnekler üzerinde tahminleri görselleştir
    """
    val_generator.reset()
    
    # Rastgele batch al
    batch_images, batch_labels = next(val_generator)
    predictions = model.predict(batch_images)
    
    # Rastgele örnekler seç
    indices = np.random.choice(len(batch_images), min(num_images, len(batch_images)), replace=False)
    
    plt.figure(figsize=(16, 12))
    for i, idx in enumerate(indices):
        plt.subplot(3, 4, i + 1)
        
        # Görüntüyü göster
        img = batch_images[idx]
        if img.max() <= 1.0:  # Normalize edilmişse
            plt.imshow(img)
        else:
            plt.imshow(img.astype('uint8'))
        
        # Gerçek ve tahmin edilen etiketler
        true_label = emotion_labels[np.argmax(batch_labels[idx])]
        pred_label = emotion_labels[np.argmax(predictions[idx])]
        confidence = np.max(predictions[idx])
        
        # Renk belirleme (doğru/yanlış)
        color = 'green' if true_label == pred_label else 'red'
        
        plt.title(f'Gerçek: {true_label.capitalize()}\n'
                 f'Tahmin: {pred_label.capitalize()}\n'
                 f'Güven: {confidence:.2f}', 
                 color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle('Örnek Tahminler (Yeşil: Doğru, Kırmızı: Yanlış)', 
                 fontsize=14, y=1.02)
    plt.show()

# Ana çalıştırma fonksiyonu
def main():
    """
    Ana program akışı
    """
    print("CNN TABANLI DUYGU ANALİZİ PROJESİ")
    print("=" * 50)
    print("Veri Seti: jonathanoheix/face-expression-recognition-dataset")
    print("Platform: Google Colab")
    print()
    
    # Veri seti bilgilerini göster
    train_counts, val_counts = get_dataset_info()
    
    print("\nModel eğitimi başlatılıyor...")
    print("Bu işlem yaklaşık 30-60 dakika sürebilir.")
    print("-" * 50)
    
    # Not: Gerçek eğitim için aşağıdaki satırları uncomment edin
    model, history, train_gen, val_gen = train_emotion_model(epochs=50, batch_size=32)
    
    # # Eğitim sonuçlarını görselleştir
    plot_training_history(history)
    
    # # Model performansını değerlendir
    accuracy, class_acc = evaluate_model(model, val_gen)
    
    # # Örnek tahminleri göster
    visualize_predictions(model, val_gen)
    
    # # Model kaydet
    model.save('emotion_recognition_model.h5')
    print("Model 'emotion_recognition_model.h5' olarak kaydedildi.")
    
    print("\nBEKLENEN SONUÇLAR (Eğitim sonrası):")
    print("-" * 40)
    print("• Genel Accuracy: ~75%")
    print("• En başarılı duygular: Happy, Sad, Neutral")
    print("• Zorluk çekilen duygular: Fear, Disgust")
    print("• Eğitim süresi: 45-60 dakika (GPU ile)")

if __name__ == "__main__":
    main()