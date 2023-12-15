import tkinter as tk
from tkinter import ttk  # Importe ttk do módulo tkinter para o combobox
from tkinter import filedialog, Label, Text
from PIL import Image, ImageTk
import numpy as np
import os
from tensorflow.keras.models import load_model

categories_8 = ["bacterialspot", "healthy", "lateblight", "leafmold", "mosaicvirus", "yellowleafcurlvirus", "spidermite", "septorialeafspot"]
categories_10 = ["bacterialspot", "healthy", "lateblight", "leafmold", "mosaicvirus", "yellowleafcurlvirus", "spidermite", "septorialeafspot", "earlyblight", "targetspot"]

def update_categories(selection):
    global categories
    if selection == '8 Classes':
        categories = categories_8
    else:
        categories = categories_10
    model_status_label.config(text=f"Categorias atualizadas para {selection}")


model = None
categories = ["bacterialspot", "healthy","lateblight","leafmold", "mosaicvirus","yellowleafcurlvirus" ,"spidermite","septorialeafspot" ]

def load_model_from_file():
    global model
    model_path = filedialog.askopenfilename(title="Selecione o Modelo", filetypes=[("Modelos Keras", "*.h5")])
    if model_path:
        try:
            model = load_model(model_path)
            model_status_label.config(text="Modelo carregado com sucesso.")
            model_name_label.config(text="Modelo: " + os.path.basename(model_path))  # Exibe o nome do modelo
        except Exception as e:
            model_status_label.config(text=f"Erro ao carregar o modelo: {e}")



def load_image():
    global img_label  # Global para atualizar o rótulo da imagem
    filepath = filedialog.askopenfilename(title="Selecione uma Imagem")
    if filepath:
        # Carregar e exibir a imagem
        img = Image.open(filepath)
        img_display = img.resize((200, 200))  # Redimensionar para exibição
        img_tk = ImageTk.PhotoImage(img_display)  # Converter para formato Tkinter
        img_label.config(image=img_tk)
        img_label.image = img_tk  # Referência de imagem para evitar garbage collection

        img = img.resize((64, 64))  # Redimensionar para o modelo
        img_array = np.array(img) / 255.0
        img_array = img_array[np.newaxis, ...]

        if model:
            predictions = model.predict(img_array)[0]
            predicted_class_index = np.argmax(predictions)
            predicted_class = categories[predicted_class_index]
            predicted_probability = predictions[predicted_class_index]

            result_prediction_text = f'Previsão: {predicted_class} ({predicted_probability * 100:.2f}%)\n\n'
            result_text = ''
            for i, category in enumerate(categories):
                result_text += f'{category}: {predictions[i] * 100:.2f}%\n'
            result_label.config(text=result_text)
            result_label_prediction.config(text=result_prediction_text)
            
        else:
            result_label.config(text="Modelo não carregado.")


def create_styled_button(parent, text, command):
    return tk.Button(parent, text=text, command=command, bg="#007bff", fg="white", height=2, width=20)

root = tk.Tk()
root.title("Modelo de Previsão de Imagem")
root.geometry("600x700")

# Frame para os botões
buttons_frame = tk.Frame(root)
buttons_frame.pack(pady=10)

# Criar botões estilizados e colocá-los no frame
load_model_button = create_styled_button(buttons_frame, "Carregar Modelo", load_model_from_file)
load_model_button.pack(side=tk.LEFT, padx=10)  # Botão à esquerda

load_image_button = create_styled_button(buttons_frame, "Carregar Imagem", load_image)
load_image_button.pack(side=tk.RIGHT, padx=10)  # Botão à direita

category_selection = ttk.Combobox(root, values=['8 Classes', '10 Classes'])
category_selection.pack(pady=5)
category_selection.bind('<<ComboboxSelected>>', lambda event: update_categories(category_selection.get()))

# Rótulo para exibir o nome do modelo
model_name_label = Label(root, text="Modelo: -----", bg="lightgrey", width=60)
model_name_label.pack(pady=10)

# Rótulo para exibir o status do modelo
model_status_label = Label(root, text="Modelo não carregado.", bg="lightgrey", width=60)
model_status_label.pack(pady=10)

img_label = Label(root)
img_label.pack(pady=10)

result_label_prediction = Label(root, text="Previsão", bg="lightgrey", width=60, height=5)
result_label_prediction.pack(pady=2)

# Rótulo para exibir o resultado
result_label = Label(root, text="Resultado da Previsão", bg="lightgrey", width=60, height=15)
result_label.pack(pady=10)


root.mainloop()