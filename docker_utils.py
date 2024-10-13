import docker
import pandas as pd
from docker.errors import BuildError, ContainerError


def get_images_and_containers():
    # Create a Docker client connected to the local Docker daemon
    client = docker.from_env()
    
    # Retrieve a list of images
    images = client.images.list()
    
    # Prepare data for DataFrame
    image_data = []
    for image in images:
        for tag in image.tags:
            repo, tag = tag.split(":") if ":" in tag else (tag, "latest")
            image_data.append({
                "REPOSITORY": repo,
                "TAG": tag,
                "IMAGE ID": image.id.split(":")[1][:12],
                "CREATED": image.attrs['Created'],
                "SIZE": image.attrs['Size']
            })
    
    # Create a DataFrame from the image data
    df1 = pd.DataFrame(image_data)
    
    containers = client.containers.list(all=True)
    
    # Prepare data for DataFrame
    container_data = []
    for container in containers:
        container_data.append({
            "CONTAINER ID": container.short_id,
            "IMAGE": container.image.tags[0] if container.image.tags else 'No tag',
            "COMMAND": container.attrs['Config']['Cmd'],
            "CREATED": container.attrs['Created'],
            "STATUS": container.status,
            "NAMES": container.name
        })
    
    # Create a DataFrame from the container data
    df2 = pd.DataFrame(container_data)
    
    return df1,df2

def run_container(image, ports):
    client = docker.from_env()
    try:
        container = client.containers.run(image, ports=ports, detach=True)
        return container.id
    except ContainerError as e:
        raise e
    
def build_image(path,tag):
    client = docker.from_env()
    try:
        image, build_logs = client.images.build(tag=tag, path=path, rm=True)
        print(build_logs)
        return image.id
    except BuildError as e:
        raise e

def stop_container(container_id):
    try:
        client = docker.from_env()
        container = client.containers.get(container_id=container_id)
        container.stop()
    except Exception as e:
        raise e

def start_container(container_id):
    try:
        client = docker.from_env()
        container = client.containers.get(container_id=container_id)
        container.start()
    except Exception as e:
        raise e
    
def delete_image(image_name):
    try:
        client = docker.from_env()
        image = client.images.get(name=image_name)
        image.remove()
    except Exception as e:
        raise e
    
def delete_container(container_id):
    try:
        client = docker.from_env()
        container = client.containers.get(container_id=container_id)
        container.remove()
    except Exception as e:
        raise e