# voiage Docker Deployment

This document describes how to deploy voiage using Docker containers for various use cases.

## Available Docker Images

1. **Web API**: RESTful API for remote VOI calculations
2. **Jupyter Notebook**: Interactive analysis environment with widgets
3. **CLI**: Command-line interface for batch processing

## Quick Start

### Web API

Run the web API service:

```bash
docker run -p 8000:8000 voiage-api
```

Or using docker-compose:

```bash
docker-compose up voiage-api
```

Access the API at `http://localhost:8000`

### Jupyter Notebook

Run the Jupyter notebook environment:

```bash
docker run -p 8888:8888 voiage-jupyter
```

Or using docker-compose:

```bash
docker-compose up voiage-jupyter
```

Access the notebook at `http://localhost:8888`

### CLI

Run the command-line interface:

```bash
docker run voiage-cli --help
```

Or using docker-compose:

```bash
docker-compose run voiage-cli --help
```

## Building Images

### Web API

```bash
docker build -t voiage-api .
```

### Jupyter Notebook

```bash
docker build -f Dockerfile.jupyter -t voiage-jupyter .
```

### CLI

```bash
docker build -f Dockerfile.cli -t voiage-cli .
```

## Full Deployment

Deploy all services using docker-compose:

```bash
docker-compose -f docker-compose.full.yml up
```

This will start:
- Web API on port 8000
- Jupyter Notebook on port 8888
- CLI service (accessible via docker-compose run)

## Data Persistence

To persist data across container restarts, mount volumes:

```bash
docker run -p 8000:8000 -v /path/to/data:/app/data voiage-api
```

## Environment Variables

The following environment variables can be set:

- `PYTHONPATH`: Python path for module discovery
- `VOIAGE_CONFIG`: Path to configuration file

## Customization

### Building with Custom Dependencies

To add custom dependencies, create a custom Dockerfile:

```dockerfile
FROM voiage-api

# Install additional packages
RUN pip install custom-package

# Copy custom code
COPY custom_module.py /app/
```

### Extending the Jupyter Environment

To add custom Jupyter extensions:

```dockerfile
FROM voiage-jupyter

# Install Jupyter extensions
RUN pip install jupyterlab-git

# Enable extensions
RUN jupyter labextension install @jupyterlab/git
```

## Security Considerations

- The default Jupyter configuration disables authentication for ease of use
- In production, enable authentication and use secure tokens
- Run containers with non-root users when possible
- Regularly update base images to include security patches

## Performance Optimization

- Use multi-stage builds to reduce image size
- Leverage Docker layer caching by ordering instructions properly
- Use `.dockerignore` to exclude unnecessary files
- Consider using DockerSlim to reduce image size

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 8000 and 8888 are available
2. **Permission errors**: Check file permissions for mounted volumes
3. **Dependency issues**: Ensure all required packages are installed

### Debugging

To debug a running container:

```bash
docker exec -it <container_id> /bin/bash
```

To view container logs:

```bash
docker logs <container_id>
```

## Maintenance

- Regularly rebuild images to include security updates
- Monitor container resource usage
- Backup important data regularly
- Test updates in a staging environment before deploying to production