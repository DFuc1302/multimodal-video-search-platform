# Dependency sets

Install only the set needed for the work:

```bash
pip install -r requirements/runtime.txt   # search API
pip install -r requirements/pipeline.txt  # preprocessing and indexing
pip install -r requirements/dev.txt       # local test environment
```

The older `requirements_wsl*.txt` files remain as pinned environment
references. They are not required to store generated data in the repository.
