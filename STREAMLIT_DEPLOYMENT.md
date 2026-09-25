# Streamlit deployment

## Files
- `app.py` — Streamlit dashboard
- `data/Potato_Price_TimeSeries_Weather_Dataset.csv` — project dataset
- `requirements.txt` — deployment dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
1. Push this repository to GitHub.
2. Open https://share.streamlit.io/
3. Sign in with GitHub and authorize Streamlit.
4. Create app.
5. Select your repository, branch `main`, and file `app.py`.
6. Deploy.
