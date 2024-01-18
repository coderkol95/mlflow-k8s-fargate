import docker
client = docker.from_env()
res= client.containers.run("ubuntu", "echo hello world")
print("Running",res)