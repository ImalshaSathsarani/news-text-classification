#Use a lightweight Python base image
FROM python:3.9-slim

#Set the working directory in the container
WORKDIR /app

#Copy requirements ans install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copy the rest of the application code
COPY . .

#Expose port 5000 for Flask
EXPOSE 5000

#Command to run the app
CMD ["python", "app.py"]