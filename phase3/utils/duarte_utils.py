import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

class DuarteUtils:

    static_image_size = 64
    
    @staticmethod
    def load_images_from_folder(categories, folder):
        from PIL import Image
        images = []
        labels = []
        subfolders = os.listdir(folder)
        for subfolder in subfolders:
            category = subfolder.split("_")[-1].lower()
            if category in categories:
                print("Loading images from category: " + category + " and subfolder: " + subfolder)
                subfolder_path = os.path.join(folder, subfolder)
                for filename in os.listdir(subfolder_path):
                    img = Image.open(os.path.join(subfolder_path, filename))
                    img = img.resize((DuarteUtils.static_image_size, DuarteUtils.static_image_size))
                    img = np.array(img) / 255.0
                    images.append(img)
                    labels.append(categories.index(category))
            else:
                print("Category: " + category + " from subfolder " + subfolder + " is not in the list of categories.")
        return np.array(images), np.array(labels)

    @staticmethod
    def split_data(x_train, y_train, reduction_train_ratio):
        x_train_split, x_reduction, y_train_split, y_reduction = train_test_split(x_train, y_train, test_size=reduction_train_ratio, random_state=42)
        print(f"Conjunto reduzido em 0.{reduction_train_ratio}")
        return x_train_split, x_reduction, y_train_split, y_reduction
        
    @staticmethod
    def display_images_data(y_test, categories):
        unique, counts = np.unique(y_test, return_counts=True)
        print("Contagem de imagens por classe: ")
        for i, category in enumerate(categories):
            print(f" - {category}: {counts[np.where(unique == i)[0][0]] if i in unique else 0}")

    @staticmethod
    def create_cnn_model(dropout_rate, learning_rate):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(DuarteUtils.static_image_size, DuarteUtils.static_image_size, 3))
    
        for layer in base_model.layers:
            layer.trainable = False
    
        # Tornar as últimas n camadas treináveis
        n_trainable_layers = 3
        for layer in base_model.layers[-n_trainable_layers:]:
            layer.trainable = True
            
        x = base_model.output
        x = Flatten()(x) 
        x = Dense(128, activation='relu')(x)  
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        predictions = Dense(len(categories), activation='softmax')(x)  # Output layer
    
        model = Model(inputs=base_model.input, outputs=predictions)
    
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    @staticmethod
    def evaluate_model(best_model, x_data, y_data):
        loss_value, test_accuracy = best_model.evaluate(x_data, y_data, verbose=0)
        print(f"acc sobre o conjunto: {test_accuracy}, Loss: {loss_value}")

    @staticmethod
    def display_roc_auc_by_class(y_test,y_pred_probs, categories):
        
        # bin the class types
        y_test_binarized = label_binarize(y_test, classes=np.arange(len(categories)))
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i, category in enumerate(categories):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plotar a curva ROC para cada classe
        plt.figure(figsize=(10, 8))
        for i, category in enumerate(categories):
            plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve of class {category} (area = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic for Each Class')
        plt.legend(loc="lower right")
        plt.show()

    @staticmethod
    def display_confusion_matrix(y_test, y_pred_classes, categories):
        conf_matrix = confusion_matrix(y_test, y_pred_classes)
        conf_matrix_percent = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix_percent, annot=True, fmt=".2%", cmap='Blues', xticklabels=categories, yticklabels=categories)
        plt.title("Matriz de Confusão (Percentagens por Classe)")
        plt.ylabel('Classe Verdadeira')
        plt.xlabel('Classe Prevista')
        plt.show()

    @staticmethod
    def display_folder_distributions(y_train, y_validation, y_test, categories):
        import numpy as np
        import matplotlib.pyplot as plt
        
        train_counts = np.bincount(y_train)
        validation_counts = np.bincount(y_validation)
        test_counts = np.bincount(y_test)
        
        max_length = len(categories)
        train_counts = np.pad(train_counts, (0, max_length - len(train_counts)), 'constant')
        validation_counts = np.pad(validation_counts, (0, max_length - len(validation_counts)), 'constant')
        test_counts = np.pad(test_counts, (0, max_length - len(test_counts)), 'constant')
        
        x = np.arange(len(categories))
        width = 0.25  # largura das barras
        
        fig, ax = plt.subplots(figsize=(15, 6))  # Gráfico mais largo
        rects1 = ax.bar(x - width, train_counts, width, label='Treino')
        rects2 = ax.bar(x, validation_counts, width, label='Validação')
        rects3 = ax.bar(x + width, test_counts, width, label='Teste')
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Número de Imagens')
        ax.set_title('Número de Imagens por Classe nos Conjuntos de Treino, Validação e Teste')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45)  # Rotação das categorias para melhorar a visualização
        ax.legend()
        
        # Função para exibir as etiquetas
        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        fig.tight_layout()
        plt.show()
