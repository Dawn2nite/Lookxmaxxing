# Selfie Coach

Selfie Coach is a lightweight browser app that gives feedback on selfie quality instead of judging appearance. It analyzes one face on-device and scores:

- lighting
- sharpness
- framing
- pose
- expression

## How to run

Because the app uses ES modules and browser camera APIs, serve the folder with a local static server:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Notes

- Face analysis runs in the browser with MediaPipe Tasks Vision loaded from CDN.
- Webcam capture requires HTTPS in production, or localhost while developing.
- The scoring is heuristic and intended for photo coaching, not identity or beauty evaluation.
