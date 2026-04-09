# ConceptPath

ConceptPath is a Flask-based prerequisite discovery platform. Users register/login, enter a target concept (or educational text), and get a generated learning roadmap with interactive multi-level graphs.

## What the project does

- User authentication (register/login/logout)
- NLP concept extraction using spaCy
- Concept lookup and relevance scoring against a local knowledge graph DB
- Graph traversal with NetworkX to build hierarchical prerequisite paths
- Interactive frontend graph visualization using Cytoscape.js
- Three separated graph views:
  - Foundational
  - Intermediate
  - Advanced
- Clickable nodes with concept info panel
- Roadmap saving and export (JSON/CSV)

## Project structure

- `app.py` - Flask backend + APIs + graph processing
- `templates/` - HTML templates (`login.html`, `register.html`, `dashboard.html`)
- `knowledge_graph.db` - graph database (concepts + relationships)
- `users.db` - user accounts + saved roadmaps
- `graph.ipynb` - notebook used to generate/populate the knowledge graph

## Run the project

See `how to run the project.txt` for exact Windows steps.

Quick start:

1. `py -3.12 -m venv .venv`
2. `.\.venv\Scripts\activate`
3. `python -m pip install -r requirements.txt`
4. `py -3.12 app.py`
5. Open `http://127.0.0.1:5000`

## Database generation (graph.ipynb)

The notebook builds `knowledge_graph.db` in a staged pipeline:

1. Install/download dependencies and datasets (`kagglehub`, spaCy model, dataset downloads).
2. Create SQLite schema:
   - `concepts(id, name, frequency)`
   - `relationships(source_id, target_id, weight, relation_type)`
3. Extract noun-phrase concepts from Coursera and ArXiv text chunks with spaCy.
4. Insert/update concepts and frequencies.
5. Build co-occurrence edges by pairing concepts that appear in the same chunk.
6. Upsert relationships and increment edge `weight` on repeated co-occurrence.
7. Verify graph population by checking relationship counts.

### Important note on data quality

In the shown notebook cells, Coursera + ArXiv ingestion is actively used for DB population. ConceptNet/Wikipedia are downloaded, but not fully ingested into relationships in the visible pipeline cells.

## API endpoints (main)

- `POST /api/get_prerequisites` - generate graph/list payload
- `POST /api/save_roadmap` - save generated roadmap for logged-in user
- `GET /api/my_roadmaps` - list saved roadmaps

## Tech stack

- Backend: Flask, NetworkX, spaCy, SQLite
- Frontend: HTML, TailwindCSS, Cytoscape.js
- Data pipeline: Jupyter Notebook (`graph.ipynb`)

## requirements.txt

`requirements.txt` in this repo was generated from environment freeze to capture exact package versions.
