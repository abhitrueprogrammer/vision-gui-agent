Known issue: 500 Internal Server Error on API requests after fresh setup

Symptom:
API endpoints (e.g. GET /brands) return an empty-body 500 with no useful error in storage/logs/laravel.log. Running the failing code path via artisan tinker surfaces the real error:

UnexpectedValueException: The stream or file "storage/logs/laravel.log" could not be
opened in append mode: Failed to open stream: Permission denied

Cause:
laravel-api's storage/ directory is bind-mounted from the host (./${SPRINT}/API:/var/www:cached in docker-compose.yml), so file ownership on storage/logs/laravel.log depends on whatever process created it on the host, not on anything set inside the Docker image. The container's PHP-FPM process runs as uid=1000 (www-data), gid=82, but the log file/directory often end up owned by a mismatched uid/gid (e.g. 82:1000) after a fresh clone or reset — leaving www-data without write access. This silently breaks Laravel's logger, which in turn breaks the whole request (not just logging).

Fix:

bash
docker compose exec -T --user root laravel-api sh -c "chown -R 1000:82 storage bootstrap/cache"

When to re-run this:

After a fresh git clone
After docker compose down -v (which recreates volumes/directories)
If you see the same Permission denied error reappear in artisan tinker output after a reset

Diagnostic tip if this happens again:
A 500 with an empty response body and nothing useful in laravel.log is a strong signal the logger itself can't write — check permissions before assuming it's an application bug:

bash
docker compose exec -T laravel-api sh -c "id && ls -la storage/logs/"

If the uid:gid from id doesn't match the file owner shown by ls -la, that's the mismatch — rerun the chown fix above.