from transformers import AutoImageProcessor, AutoModelForImageClassification

# Nombre del modelo en Hugging Face
model_name = "microsoft/resnet-18"

# Descargar el procesador de imágenes
# Este objeto se encargará de normalizar y transformar las imágenes
image_processor = AutoImageProcessor.from_pretrained(model_name)

# Descargar el modelo preentrenado
# Este objeto es la arquitectura ResNet-18 con los pesos preentrenados
model = AutoModelForImageClassification.from_pretrained(model_name)

print("Modelo y procesador de imágenes descargados correctamente.")
print("\nDetalles del modelo:")
print(model)
print("\nDetalles del procesador de imágenes:")
print(image_processor)