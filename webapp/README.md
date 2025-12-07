# Refrag Webapp

This folder contains a minimal Flask server and a React (Vite) client that implement a side-by-side chat demo.

Server
------
Requirements: `pip install -r ../requirements.txt` (make sure to activate your venv)

Run the server:

```cmd
cd webapp
python server.py
```

This runs on http://0.0.0.0:7860 and serves two endpoints:
- `GET /models` - returns a list of available model names
- `POST /chat` - accepts JSON {"prompt":..., "model":...} and returns a text/event-stream of token objects

Client
------
Client is a Vite + React app in `webapp/client`.

Install and run:

```cmd
cd webapp\client
npm install
npm run dev
```

Open the app at the URL printed by Vite (usually http://localhost:5173). The React app expects the Flask server at `http://localhost:7860`.

Notes
-----
- The current server uses a fake streaming implementation. Later we'll replace it with a real model-backed streamer that interfaces with transformers or your Refrag model.
- Model selection is persisted in `localStorage` (leftModel/rightModel keys).
- The client measures simple token timing metrics and updates them as tokens arrive.

