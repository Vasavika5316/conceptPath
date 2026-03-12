from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
import json
import re
from datetime import datetime
from difflib import SequenceMatcher

import networkx as nx
import spacy

app = Flask(__name__)
app.secret_key = 'conceptpath_secret'

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = spacy.blank("en")

GRAPH_CACHE = {
    "graph": None,
    "id_to_name": {},
    "name_to_id": {},
    "concept_tokens": {},
    "token_to_ids": {},
}

NOISY_TOKENS = {
    "course", "class", "student", "lesson", "chapter", "video", "part", "topic", "overview",
    "introduction", "basics", "basic", "advanced", "important", "easy", "simple", "section"
}


def init_user_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS roadmaps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            roadmap_name TEXT NOT NULL,
            target_concept TEXT NOT NULL,
            input_text TEXT,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(username, roadmap_name)
        )
    ''')

    conn.commit()
    conn.close()


def get_user_db():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn


def load_knowledge_graph():
    if GRAPH_CACHE["graph"] is not None:
        return (
            GRAPH_CACHE["graph"],
            GRAPH_CACHE["id_to_name"],
            GRAPH_CACHE["name_to_id"],
            GRAPH_CACHE["concept_tokens"],
            GRAPH_CACHE["token_to_ids"],
        )

    conn = sqlite3.connect('knowledge_graph.db')
    cursor = conn.cursor()

    cursor.execute("SELECT id, name FROM concepts")
    concepts = cursor.fetchall()

    cursor.execute("SELECT source_id, target_id, weight FROM relationships")
    edges = cursor.fetchall()
    conn.close()

    graph = nx.Graph()
    id_to_name = {}
    name_to_id = {}
    concept_tokens = {}
    token_to_ids = {}

    for concept_id, name in concepts:
        normalized = name.strip()
        id_to_name[concept_id] = normalized
        if normalized.lower() not in name_to_id:
            name_to_id[normalized.lower()] = concept_id

        tokens = tokenize_text(normalized)
        concept_tokens[concept_id] = tokens
        for token in tokens:
            token_to_ids.setdefault(token, set()).add(concept_id)

        graph.add_node(concept_id)

    for source_id, target_id, weight in edges:
        graph.add_edge(source_id, target_id, weight=weight or 1)

    GRAPH_CACHE["graph"] = graph
    GRAPH_CACHE["id_to_name"] = id_to_name
    GRAPH_CACHE["name_to_id"] = name_to_id
    GRAPH_CACHE["concept_tokens"] = concept_tokens
    GRAPH_CACHE["token_to_ids"] = token_to_ids
    return graph, id_to_name, name_to_id, concept_tokens, token_to_ids


def tokenize_text(text):
    def normalize_token(token):
        if token.endswith('s') and len(token) > 4:
            return token[:-1]
        return token

    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    normalized = [normalize_token(t) for t in tokens]
    return [t for t in normalized if len(t) > 2 and t not in NOISY_TOKENS]


def extract_candidate_concepts(text, limit=8):
    raw = (text or "").strip().lower()
    if not raw:
        return []

    candidates = [raw] if len(raw.split()) <= 6 else []
    doc = nlp(raw)

    if getattr(doc, "ents", None):
        for ent in doc.ents:
            phrase = ent.text.strip().lower()
            if len(phrase) > 2:
                candidates.append(phrase)

    if hasattr(doc, "noun_chunks"):
        try:
            for chunk in doc.noun_chunks:
                phrase = " ".join(
                    token.lemma_.lower()
                    for token in chunk
                    if not token.is_stop and token.is_alpha
                ).strip()
                if len(phrase) > 2:
                    candidates.append(phrase)
        except Exception:
            pass

    for token in doc:
        if token.is_alpha and not token.is_stop and token.pos_ in {"NOUN", "PROPN"}:
            lemma = token.lemma_.lower().strip()
            if len(lemma) > 2:
                candidates.append(lemma)

    seen = set()
    unique = []
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)

    return unique[:limit]


def concept_match_score(candidate_text, candidate_tokens, concept_name, concept_tokens):
    if not candidate_text:
        return -1.0

    concept_name_l = concept_name.lower()
    concept_token_set = set(concept_tokens)
    candidate_token_set = set(candidate_tokens)

    exact_score = 100.0 if candidate_text == concept_name_l else 0.0
    phrase_score = 0.0
    if candidate_text in concept_name_l:
        phrase_score = 55.0 + min(20.0, len(candidate_text) / max(1.0, len(concept_name_l)) * 20.0)

    overlap = len(candidate_token_set & concept_token_set)
    union = max(1, len(candidate_token_set | concept_token_set))
    jaccard_score = (overlap / union) * 45.0
    containment_score = 0.0
    if candidate_token_set and candidate_token_set.issubset(concept_token_set):
        containment_score = 20.0

    similarity_score = SequenceMatcher(None, candidate_text, concept_name_l).ratio() * 20.0

    degree_penalty = max(0.0, (len(concept_tokens) - 4) * 1.2)
    specificity_penalty = 0.0
    if len(candidate_token_set) >= 2 and len(concept_token_set) == 1:
        specificity_penalty = 35.0
    return exact_score + phrase_score + jaccard_score + containment_score + similarity_score - degree_penalty - specificity_penalty


def resolve_concept_id(candidates, id_to_name, name_to_id, concept_tokens, token_to_ids):
    if not candidates:
        return None

    best_id = None
    best_score = -1.0

    first_candidate = (candidates[0] or "").strip().lower() if candidates else ""
    first_tokens = tokenize_text(first_candidate)
    if first_candidate in name_to_id:
        return name_to_id[first_candidate]

    for index, candidate in enumerate(candidates):
        candidate_text = (candidate or "").strip().lower()
        if not candidate_text:
            continue

        candidate_tokens = tokenize_text(candidate_text)
        if not candidate_tokens:
            candidate_tokens = [candidate_text]

        if index > 0 and len(first_tokens) >= 2 and len(candidate_tokens) == 1:
            continue

        token_pools = [token_to_ids.get(token, set()) for token in candidate_tokens if token_to_ids.get(token)]
        candidate_pool = set()
        if len(candidate_tokens) >= 2 and token_pools:
            intersection_pool = set(token_pools[0])
            for pool in token_pools[1:]:
                intersection_pool &= pool
            if intersection_pool:
                candidate_pool = intersection_pool

        if not candidate_pool:
            for pool in token_pools:
                candidate_pool.update(pool)

        if not candidate_pool:
            candidate_pool = set(id_to_name.keys())

        for concept_id in candidate_pool:
            name = id_to_name.get(concept_id, "")
            score = concept_match_score(
                candidate_text,
                candidate_tokens,
                name,
                concept_tokens.get(concept_id, []),
            )
            if index == 0:
                score += 12.0
            score -= index * 2.5
            if score > best_score:
                best_score = score
                best_id = concept_id

    return best_id if best_score >= 32 else None


def weighted_relevance(weight, degree):
    return float(weight) / (1.0 + 0.02 * max(1, degree))


def expand_relevant_frontiers(graph, start_id, depth, per_level_limit, concept_tokens):
    safe_limit = max(4, min(int(per_level_limit or 18), 50))
    visited = {start_id}
    frontier = [start_id]
    levels = {1: [], 2: [], 3: []}
    parent_of = {}
    score_of = {start_id: 100.0}
    anchor_tokens = set(concept_tokens.get(start_id, []))

    for distance in range(1, depth + 1):
        candidates = {}
        for parent in frontier:
            for neighbor in graph.neighbors(parent):
                if neighbor in visited:
                    continue

                edge_weight = graph[parent][neighbor].get('weight', 1)
                score = weighted_relevance(edge_weight, graph.degree(neighbor))

                neighbor_tokens = set(concept_tokens.get(neighbor, []))
                overlap = len(anchor_tokens & neighbor_tokens)
                if overlap:
                    score *= 1.0 + (0.8 * overlap)
                elif anchor_tokens:
                    if distance == 1:
                        score *= 0.2
                    elif distance == 2:
                        score *= 0.45
                    else:
                        score *= 0.65

                existing = candidates.get(neighbor)
                if existing is None or score > existing[0]:
                    candidates[neighbor] = (score, parent, overlap)

        if distance <= 2:
            overlap_candidates = {
                node_id: data for node_id, data in candidates.items() if data[2] > 0
            }
            if overlap_candidates:
                candidates = overlap_candidates

        ranked = sorted(candidates.items(), key=lambda item: (-item[1][0], item[0]))
        selected_nodes = []
        for node_id, (score, parent, overlap) in ranked[:safe_limit]:
            visited.add(node_id)
            selected_nodes.append(node_id)
            parent_of[node_id] = parent
            score_of[node_id] = round(score, 4)

        bucket = 3 if distance >= 3 else distance
        levels[bucket].extend(selected_nodes)
        frontier = selected_nodes

        if not frontier:
            break

    return levels, parent_of, score_of

    return None


def level_from_distance(distance):
    if distance >= 3:
        return "Foundational"
    if distance == 2:
        return "Intermediate"
    return "Advanced"


def build_prerequisite_payload(graph, id_to_name, concept_tokens, start_id, source_text, depth=3, per_level_limit=18):
    if start_id not in graph:
        return None

    tiered_nodes, parent_of, score_of = expand_relevant_frontiers(
        graph,
        start_id,
        depth,
        per_level_limit,
        concept_tokens,
    )

    selected = [start_id]
    selected.extend(tiered_nodes[3])
    selected.extend(tiered_nodes[2])
    selected.extend(tiered_nodes[1])

    if len(selected) == 1:
        return None

    distances = {start_id: 0}
    for bucket, nodes in tiered_nodes.items():
        distance = 3 if bucket == 3 else bucket
        for node_id in nodes:
            distances[node_id] = distance

    subgraph = graph.subgraph(selected)
    directed = nx.DiGraph()
    directed.add_nodes_from(selected)

    for node_id, parent in parent_of.items():
        if node_id in subgraph and parent in subgraph:
            directed.add_edge(node_id, parent)

    levels = {"Foundational": [], "Intermediate": [], "Advanced": []}
    learning_path = []
    node_data = []

    for node_id in selected:
        name = id_to_name.get(node_id, str(node_id))
        distance = distances.get(node_id, 0)
        level = level_from_distance(distance)
        is_target = node_id == start_id

        if not is_target:
            levels[level].append(name)

        node_data.append({
            "data": {
                "id": str(node_id),
                "label": name,
                "level": level,
                "distance": distance,
                "is_target": is_target,
                "relevance": score_of.get(node_id, 100.0 if is_target else 0.0),
            }
        })

    sorted_path_ids = sorted(
        selected,
        key=lambda node_id: (
            -distances.get(node_id, 0),
            -score_of.get(node_id, 0.0),
            node_id,
        ),
    )
    for idx, node_id in enumerate(sorted_path_ids, start=1):
        learning_path.append({
            "step": idx,
            "concept": id_to_name.get(node_id, str(node_id)),
            "level": level_from_distance(distances.get(node_id, 0)),
            "distance": distances.get(node_id, 0),
        })

    edge_data = [
        {"data": {"source": str(source), "target": str(target)}}
        for source, target in directed.edges()
    ]

    return {
        "nodes": node_data,
        "edges": edge_data,
        "list": levels,
        "learning_path": learning_path,
        "target": id_to_name.get(start_id, ""),
        "input_text": source_text,
        "focused_node_id": str(start_id),
    }


def get_graph_data(target_text=None, start_node_id=None, node_limit=18):
    graph, id_to_name, name_to_id, concept_tokens, token_to_ids = load_knowledge_graph()

    if start_node_id is not None:
        try:
            start_id = int(start_node_id)
        except (TypeError, ValueError):
            start_id = None

        if start_id is None or start_id not in graph:
            return {
                "nodes": [],
                "edges": [],
                "list": {"Foundational": [], "Intermediate": [], "Advanced": []},
                "learning_path": [],
                "target": "",
            }

        payload = build_prerequisite_payload(
            graph,
            id_to_name,
            concept_tokens,
            start_id,
            target_text or "",
            per_level_limit=node_limit,
        )
        return payload or {
            "nodes": [],
            "edges": [],
            "list": {"Foundational": [], "Intermediate": [], "Advanced": []},
            "learning_path": [],
            "target": "",
        }

    candidates = extract_candidate_concepts(target_text)
    start_id = resolve_concept_id(candidates, id_to_name, name_to_id, concept_tokens, token_to_ids)

    if start_id is None:
        fallback = candidates[0] if candidates else (target_text or "").strip().lower()
        return {
            "nodes": [],
            "edges": [],
            "list": {"Foundational": [], "Intermediate": [], "Advanced": []},
            "learning_path": [],
            "target": fallback,
        }

    payload = build_prerequisite_payload(
        graph,
        id_to_name,
        concept_tokens,
        start_id,
        target_text or "",
        per_level_limit=node_limit,
    )
    return payload or {
        "nodes": [],
        "edges": [],
        "list": {"Foundational": [], "Intermediate": [], "Advanced": []},
        "learning_path": [],
        "target": "",
    }


@app.route('/')
def index():
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/register', methods=['GET','POST'])
def register():
    error = None

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_user_db()

        try:
            conn.execute(
                'INSERT INTO users (username,password) VALUES (?,?)',
                (username,password)
            )
            conn.commit()
            conn.close()
            return redirect(url_for('login'))

        except sqlite3.IntegrityError:
            error = "Username already exists. Try another."
            conn.close()

    return render_template('register.html', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_user_db()
        user = conn.execute(
            'SELECT * FROM users WHERE username=? AND password=?',
            (username, password)
        ).fetchone()
        conn.close()

        if user:
            session['username'] = user['username']
            return redirect(url_for('dashboard'))

        error = "Invalid username or password"

    return render_template('login.html', error=error)

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])


@app.route('/api/get_prerequisites', methods=['POST'])
def get_prerequisites():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    text_input = data.get('concept', '')
    node_id = data.get('node_id')
    node_limit = data.get('node_limit', 18)
    result = get_graph_data(text_input, node_id, node_limit)
    return jsonify(result)


@app.route('/api/save_roadmap', methods=['POST'])
def save_roadmap():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}

    roadmap_name = (data.get('roadmap_name') or '').strip()
    target = (data.get('target') or '').strip()
    input_text = data.get('input_text') or ''
    roadmap_payload = data.get('roadmap') or {}

    if not roadmap_name:
        return jsonify({"error": "Roadmap name required"}), 400

    if not target or not roadmap_payload:
        return jsonify({"error": "Missing roadmap data"}), 400

    conn = get_user_db()

    # check duplicates and auto-rename: "Name", "Name (2)", "Name (3)", ...
    base_name = roadmap_name
    counter = 1

    while True:
        existing = conn.execute(
            'SELECT id FROM roadmaps WHERE username=? AND roadmap_name=?',
            (session['username'], roadmap_name)
        ).fetchone()

        if not existing:
            break

        counter += 1
        roadmap_name = f"{base_name} ({counter})"

    conn.execute(
        '''INSERT INTO roadmaps 
        (username, roadmap_name, target_concept, input_text, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)''',
        (
            session['username'],
            roadmap_name,
            target,
            input_text,
            json.dumps(roadmap_payload),
            datetime.utcnow().isoformat(timespec='seconds') + 'Z'
        )
    )

    conn.commit()
    conn.close()

    return jsonify({
        "status": "saved",
        "roadmap_name": roadmap_name
    })


@app.route('/api/my_roadmaps', methods=['GET'])
def my_roadmaps():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    conn = get_user_db()
    rows = conn.execute(
        'SELECT id, roadmap_name, target_concept, created_at FROM roadmaps WHERE username = ? ORDER BY id DESC LIMIT 20',
        (session['username'],),
    ).fetchall()
    conn.close()

    return jsonify({
        "items": [
            {
                "id": row['id'],
                "target": row['target_concept'],
                "name": row['roadmap_name'],
                "created_at": row['created_at'],
            }
            for row in rows
        ]
    })


@app.route('/api/get_roadmap/<int:roadmap_id>', methods=['GET'])
def get_roadmap(roadmap_id):
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    conn = get_user_db()
    row = conn.execute(
        'SELECT payload_json FROM roadmaps WHERE id = ? AND username = ?',
        (roadmap_id, session['username'])
    ).fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Not found"}), 404

    return jsonify(json.loads(row['payload_json']))


@app.route('/api/delete_roadmap/<int:roadmap_id>', methods=['POST'])
def delete_roadmap(roadmap_id):
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    conn = get_user_db()
    cursor = conn.cursor()

    cursor.execute(
        'DELETE FROM roadmaps WHERE id = ? AND username = ?',
        (roadmap_id, session['username'])
    )
    deleted = cursor.rowcount

    conn.commit()
    conn.close()

    if deleted == 0:
        return jsonify({"error": "Not found"}), 404

    return jsonify({"status": "deleted", "id": roadmap_id})

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


if __name__ == '__main__':
    init_user_db()
    app.run(debug=True)

