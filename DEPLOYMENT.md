# Deploy on Streamlit Community Cloud

## 1. Push code to GitHub

- Do **not** push class recordings, `.env`, or `.streamlit/secrets.toml`.
- Main entry file: **`app.py`**
- Dependencies: **`requirements.txt`**
- System package: **`packages.txt`** (installs `ffmpeg`)

## 2. Create the app

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → select your repo, branch `main`, main file **`app.py`**.
3. Open **Advanced settings** if needed (defaults are usually fine).

## 3. Secrets (required for hosted LLM)

Streamlit Cloud **cannot** use local Ollama. In **Settings → Secrets**, paste (see `.streamlit/secrets.toml.example`):

```toml
LLM_PROVIDER = "openai"
OPENAI_API_KEY = "sk-your-key"
OPENAI_MODEL = "gpt-4o-mini"
```

For local open-source runs, keep using Ollama via `.env` on your machine.

Optional — show the public link inside the app:

```toml
DEPLOYED_APP_URL = "https://YOUR-APP-NAME.streamlit.app"
```

Use the exact URL Streamlit gives you after the first successful deploy.

## 4. Deploy and copy your link

When the app is **Running**, copy the URL, e.g.:

`https://my-class-quality-assessment.streamlit.app`

## 5. Add the link to the README

Edit **`README.md`** → section **Live demo** → replace the placeholder:

```markdown
## Live demo

**[Open the hosted app](https://YOUR-APP-NAME.streamlit.app)**
```

Commit and push so judges see the link on GitHub.

## 6. Challenge submission

Paste the same URL in the Google Form as your **working application** link.

## Tips

- Ask reviewers to upload **MP4 + VTT** when possible (Whisper on free tier is slow for long classes).
- Test in an incognito window after deploy.
- If `ffmpeg` errors appear, confirm `packages.txt` is in the repo root.
