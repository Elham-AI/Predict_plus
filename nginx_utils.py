import os

def add_model_to_nginx_config(user_id,model_name, container_port):
    """
    Update the Nginx configuration to route traffic to a deployed Docker container.

    Parameters:
    - domain_name (str): The domain name for the application (e.g., "example.com").
    - model_name (str): The name of the model (used to define the path).
    - container_port (int): The port where the model's Docker container is exposed.

    Returns:
    - None
    """
    # Define the path for the Nginx config file
    nginx_config_file = "/etc/nginx/sites-available/elham.ai"
    os.system(f"sudo chmod 666 {nginx_config_file}")
    # Create the upstream block for the new model
    upstream_block = f"""
    upstream {user_id}_{model_name}_upstream {{
        server 127.0.0.1:{container_port};
    }}
    """
    
    # Create the location block for the new model path
    location_block = f"""
    location  /{user_id}/{model_name} {{
        proxy_pass http://{user_id}_{model_name}_upstream/{user_id}/{model_name};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
    """
    

    with open(nginx_config_file, "r") as file:
        nginx_config = file.read()


    # Check if the model is already configured
    if f"upstream {user_id}_{model_name}_upstream" in nginx_config:
        print(f"The model '{user_id}_{model_name}' is already configured in the Nginx config.")
        return

    # Insert the new upstream and location blocks
    nginx_config = f"{upstream_block}\n\n{nginx_config}"
    # nginx_config = nginx_config.replace(
    #     f"server_name {domain_name};",
    #     f"server_name {domain_name};\n\n{upstream_block}"
    # )
    nginx_config = nginx_config.replace(
        f"#<add_block_here>",
        f"#<add_block_here>\n\n{location_block}"
    )
    
    # Write the updated Nginx config back to the file
    with open(nginx_config_file, "w") as file:
        file.write(nginx_config)
    
    # Reload Nginx to apply changes
    os.system("sudo nginx -s reload")
    os.system(f"sudo chmod 644 {nginx_config_file}")
    print(f"Nginx configuration updated and reloaded for model '{model_name}'.")

def delete_model_from_nginx_config(user_id,model_name, container_port):
    """
    Update the Nginx configuration to route traffic to a deployed Docker container.

    Parameters:
    - domain_name (str): The domain name for the application (e.g., "example.com").
    - model_name (str): The name of the model (used to define the path).
    - container_port (int): The port where the model's Docker container is exposed.

    Returns:
    - None
    """
    # Define the path for the Nginx config file
    nginx_config_file = "/etc/nginx/sites-available/elham.ai"
    os.system(f"sudo chmod 666 {nginx_config_file}")
    # Create the upstream block for the new model
    upstream_block = f"""
    upstream {user_id}_{model_name}_upstream {{
        server 127.0.0.1:{container_port};
    }}
    """
    
    # Create the location block for the new model path
    location_block = f"""
    location /{user_id}/{model_name} {{
        proxy_pass http://{user_id}_{model_name}_upstream/{user_id}/{model_name};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
    """
    
    with open(nginx_config_file, "r") as file:
        nginx_config = file.read()


    # Delete the new upstream and location blocks
    nginx_config = nginx_config.replace(
        upstream_block,
        ""
    )
    nginx_config = nginx_config.replace(
        location_block,
        ""
    )
    
    # Write the updated Nginx config back to the file
    with open(nginx_config_file, "w") as file:
        file.write(nginx_config)
    
    # Reload Nginx to apply changes
    os.system(f"sudo chmod 644 {nginx_config_file}")
    os.system("sudo nginx -s reload")
    print(f"Nginx configuration updated and reloaded for model '{model_name}'.")
