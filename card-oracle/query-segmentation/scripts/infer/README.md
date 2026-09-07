# Planned inference entrypoint

segment.py will support --query, interactive, and JSONL batch modes. Load the saved
artifact once and return original text, typed character spans, and uncalibrated
confidence scores. Logs go to stderr; predictions go to stdout. No entity linking
or model implementation yet.
