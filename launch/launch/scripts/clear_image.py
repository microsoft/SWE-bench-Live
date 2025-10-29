import docker
import json
from fire import Fire
from docker.errors import ImageNotFound

def main(dataset: str):
    client = docker.from_env()
    with open(dataset, "r") as f:
        instances = [json.loads(line) for line in f]
    
    for instance in instances:
        image_name = instance["docker_image"]
        instance_id = instance['instance_id']
        
        try:
            # Pre-check if image exists
            image = client.images.get(image_name)
            print(f"clearing {instance_id} -- {image_name}")
            client.images.remove(image_name)
        except ImageNotFound:
            print(f"Image {image_name} not found for instance {instance_id}, skipping...")
        except Exception as e:
            print(f"Error processing {image_name} for instance {instance_id}: {e}")

if __name__ == "__main__":
    Fire(main)