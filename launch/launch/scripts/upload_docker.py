from fire import Fire
import json
import docker
from rich.console import Console
from rich.progress import Progress
from rich import print as rprint

def main(dataset: str,
        clear_after_push: str):
    console = Console()
    client = docker.from_env()
    
    with open(dataset, "r") as f:
        instances = [json.loads(line) for line in f]

    existing_images = set()
    for img in client.images.list():
        if img.tags:
            existing_images.update(img.tags)
    
    with Progress() as progress:
        task = progress.add_task("Pushing images...", total=len(instances))
        
        for instance in instances:
            image_key = instance["docker_image"]
            
            if image_key not in existing_images:
                rprint(f"[yellow]Warning: Image {image_key} not found locally[/yellow]")
                progress.advance(task)
                continue
            
            try:
                rprint(f"[blue]Pushing {image_key}[/blue]")
                resp = client.images.push(image_key)
                # print(resp)
                for line in resp:                 # look for {"error": "..."}
                    if "error" in line:
                        raise RuntimeError(line["error"])
                rprint(f"[green]Successfully pushed {image_key}[/green]")
            except Exception as e:
                rprint(f"[red]Error pushing {image_key}: {str(e)}[/red]")
            try:
                if int(clear_after_push):
                    client.images.remove(image_key)
                    rprint(f"[green]Successfully cleared {image_key}[/green]")
            except Exception as e:
                rprint(f"[red]Error clearing {image_key}: {str(e)}[/red]")
            
            progress.advance(task)

if __name__ == "__main__":
    Fire(main)
