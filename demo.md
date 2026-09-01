1. `uv run visual-function-lab --port 4200`

2. 
```bash
uv run vision-gui-agent http://127.0.0.1:4200 \
    "Open the document and export it as PDF" \
    --headed \
    --memory-mode none \
    --max-steps 12 \
    --artifacts artifacts/manual-local \
    --verbose \
    --model gemini-flash-lite-latest
    ```

3. 
```bash


```