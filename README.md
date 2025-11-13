# Bachelorarbeit

This project uses `uv` for dependency management.

## Setup

1. **Install `uv`**

   If you don't have `uv` installed, you can install it with:

   ```sh
   pip install uv
   ```

2. **Create a virtual environment**

   Create a virtual environment in the project root:

   ```sh
   uv venv
   ```

3. **Activate the virtual environment**

   Activate the virtual environment:

   - On macOS and Linux:
     ```sh
     source .venv/bin/activate
     ```
   - On Windows:
     ```sh
     .venv\Scripts\activate
     ```

4. **Install dependencies**

   Install the project dependencies with `uv`:

   ```sh
   uv sync
   ```

## Running the application

To run the Streamlit application, use the following command:

```sh
uv run streamlit run app.py
```
