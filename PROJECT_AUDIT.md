# Project Audit & Changes

## Original gaps found

1. The notebook depended on `Agri_Commodity_Price_Dataset_2000rows.xlsx`, but that file was not present in the repository.
2. The original notebook rebuilt the supplied time-series dataset during execution instead of treating the saved CSV/XLSX as the reproducible input.
3. The chatbot did not preserve state/season context between turns.
4. The RAG section was retrieval + templates but did not use a vector retrieval layer.
5. The README described a GitHub Pages workflow that was missing from the repository.
6. The data-provenance limitation needed to be made more prominent so simulated historical/weather values are not presented as genuine observations.

## Changes made

- Rewrote the notebook to load the included CSV directly.
- Added explicit dataset validation and provenance notes.
- Kept chronological 8-week holdout evaluation.
- Kept model comparison, Random Forest hyperparameter sweep, MAE/RMSE/R² and GridSearchCV.
- Added documented prompt-based extraction with six assignment prompts.
- Added TF-IDF/cosine-similarity retrieval documents for the RAG assistant.
- Added conversational context for state and season follow-ups.
- Added dashboard summary export to `outputs/dashboard_summary.json`.
- Updated the static dashboard chatbot to remember state/season context.
- Added an academic-data disclaimer to the dashboard.
- Added a working GitHub Pages deployment workflow.
- Updated README to map every assignment requirement to an implementation.

## How to describe the RAG in viva

> “The chatbot first retrieves relevant data-derived documents using TF-IDF and cosine similarity. It then augments the retrieval with the previous conversation context, such as the state or season, and generates a natural-language response from the retrieved statistics. It is a lightweight RAG implementation without an external LLM API.”

## How to describe the data limitation

> “The prepared 52-week history and weather variables are documented simulation anchored to a real market snapshot. Therefore, the reported metrics demonstrate the methodology on the supplied academic dataset, not production forecasting accuracy. Real deployment would require genuine multi-year mandi and weather observations.”
