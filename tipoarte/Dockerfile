#Alpine creates a lighter image of the OS.
FROM python:3.7-slim

# Copy the contents of the current directory insie the docker image
ADD . /app

# Set the home of the docker image
WORKDIR /app.py

# Install the requirements
RUN pip3 install -r requirements.txt

# Do not run the container in superuser mode
RUN adduser --disabled-login myuser
USER myuser

# Command to run when starting the container
CMD ["python3","-u","app.py"]




