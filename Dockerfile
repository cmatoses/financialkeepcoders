# Usa una imagen base oficial de Python
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo requirements.txt al directorio de trabajo
COPY requirements.txt .

# Instala las dependencias
RUN pip install -r requirements.txt

# Copia el resto del código al directorio de trabajo
COPY . .

# Expone el puerto en el que la aplicación escuchará (ajusta esto si es necesario)
EXPOSE 8080

# Define el comando para ejecutar la aplicación
CMD ["/bin/bash"]