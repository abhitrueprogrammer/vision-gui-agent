1. `uv run visual-function-lab --port 4200`

2. //old
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
uv run vision-gui-agent http://127.0.0.1:4200/fullsuite \
    "export the Launch Brief as PDF" \
    --headed \
    --memory-mode active-action-model \
    --max-steps 20 \
    --artifacts artifacts/141 \
    --verbose \
    --model gemini-flash-lite-latest
```
4. 
```bash
uv run vision-gui-agent https://wikipedia.com \
    "Open article on Abhinav Bindra" \
    --headed \
    --memory-mode none \
    --max-steps 20 \
    --artifacts artifacts/141 \
    --verbose \
    --model gemini-flash-lite-latest
```
5. 
```bash
curl -s "http://127.0.0.1:4200/reset?state=blank&layout=classic"
uv run vision-gui-agent http://127.0.0.1:4200/fullsuite \
    "create a Q3 Forecast report" \
    --headed \
    --memory-mode active-action-model \
    --max-steps 20 \
    --artifacts artifacts/142 \
    --verbose \
    --model gemini-flash-lite-latest
```
6. 
```bash
curl -s "http://127.0.0.1:4200/reset?state=blank&layout=classic"
uv run vision-gui-agent http://127.0.0.1:4200/fullsuite \
    "sign in and enable project approval with required reviewers" \
    --headed \
    --memory-mode active-action-model \
    --max-steps 20 \
    --artifacts artifacts/143 \
    --verbose \`
    --model gemini-flash-lite-latest
``