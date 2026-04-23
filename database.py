import sqlite3
import hashlib
import os
from datetime import datetime

DB_PATH = "musicface.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't already exist."""
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT    UNIQUE NOT NULL,
            password TEXT    NOT NULL,
            created  TEXT    NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS emotion_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            emotion    TEXT    NOT NULL,
            detected   TEXT    NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS recommended_songs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            track_name TEXT    NOT NULL,
            artists    TEXT    NOT NULL,
            recommended TEXT   NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS liked_songs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            track_name TEXT    NOT NULL,
            artists    TEXT    NOT NULL,
            liked      TEXT    NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ---------------------------------------------------------------------------
# User operations
# ---------------------------------------------------------------------------

def register_user(username, password):
    """
    Returns (True, user_id) on success.
    Returns (False, error_message) on failure.
    """
    conn = get_connection()
    c    = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, password, created) VALUES (?, ?, ?)",
            (username.strip(), hash_password(password), datetime.now().isoformat())
        )
        conn.commit()
        user_id = c.lastrowid
        return True, user_id
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please choose another."
    finally:
        conn.close()


def login_user(username, password):
    """
    Returns (True, user_id) if credentials are correct.
    Returns (False, error_message) otherwise.
    """
    conn = get_connection()
    c    = conn.cursor()
    c.execute(
        "SELECT id, password FROM users WHERE username = ?",
        (username.strip(),)
    )
    row = c.fetchone()
    conn.close()

    if row is None:
        return False, "Username not found."
    if row["password"] != hash_password(password):
        return False, "Incorrect password."
    return True, row["id"]


def get_username(user_id):
    conn = get_connection()
    c    = conn.cursor()
    c.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    return row["username"] if row else "User"


def update_username(user_id, new_username):
    conn = get_connection()
    c    = conn.cursor()
    try:
        c.execute(
            "UPDATE users SET username = ? WHERE id = ?",
            (new_username.strip(), user_id)
        )
        conn.commit()
        return True, "Username updated successfully."
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    finally:
        conn.close()


def update_password(user_id, new_password):
    conn = get_connection()
    c    = conn.cursor()
    c.execute(
        "UPDATE users SET password = ? WHERE id = ?",
        (hash_password(new_password), user_id)
    )
    conn.commit()
    conn.close()
    return True, "Password updated successfully."


# ---------------------------------------------------------------------------
# Emotion history
# ---------------------------------------------------------------------------

def log_emotion(user_id, emotion):
    conn = get_connection()
    c    = conn.cursor()
    c.execute(
        "INSERT INTO emotion_history (user_id, emotion, detected) VALUES (?, ?, ?)",
        (user_id, emotion, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_emotion_counts(user_id):
    """Returns dict of {emotion: count} for all emotions detected for this user."""
    conn = get_connection()
    c    = conn.cursor()
    c.execute(
        "SELECT emotion, COUNT(*) as count FROM emotion_history WHERE user_id = ? GROUP BY emotion",
        (user_id,)
    )
    rows = c.fetchall()
    conn.close()
    return {row["emotion"]: row["count"] for row in rows}


def get_most_detected_emotion(user_id):
    counts = get_emotion_counts(user_id)
    if not counts:
        return None
    return max(counts, key=counts.get)


# ---------------------------------------------------------------------------
# Recommended song history (to avoid repeats)
# ---------------------------------------------------------------------------

def log_recommended_songs(user_id, songs):
    """Store every recommended song so they won't be shown again."""
    conn = get_connection()
    c    = conn.cursor()
    for s in songs:
        c.execute(
            "INSERT INTO recommended_songs (user_id, track_name, artists, recommended) VALUES (?, ?, ?, ?)",
            (user_id, s["track_name"], s["artists"], datetime.now().isoformat())
        )
    conn.commit()
    conn.close()


def get_seen_songs(user_id):
    """Returns a set of (track_name, artists) tuples the user has already seen."""
    conn = get_connection()
    c    = conn.cursor()
    c.execute(
        "SELECT track_name, artists FROM recommended_songs WHERE user_id = ?",
        (user_id,)
    )
    rows = c.fetchall()
    conn.close()
    return {(row["track_name"], row["artists"]) for row in rows}


# ---------------------------------------------------------------------------
# Liked songs
# ---------------------------------------------------------------------------

def like_song(user_id, track_name, artists):
    conn = get_connection()
    c    = conn.cursor()
    # Avoid duplicate likes
    c.execute(
        "SELECT id FROM liked_songs WHERE user_id = ? AND track_name = ? AND artists = ?",
        (user_id, track_name, artists)
    )
    if c.fetchone():
        conn.close()
        return
    c.execute(
        "INSERT INTO liked_songs (user_id, track_name, artists, liked) VALUES (?, ?, ?, ?)",
        (user_id, track_name, artists, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def unlike_song(user_id, track_name, artists):
    conn = get_connection()
    c    = conn.cursor()
    c.execute(
        "DELETE FROM liked_songs WHERE user_id = ? AND track_name = ? AND artists = ?",
        (user_id, track_name, artists)
    )
    conn.commit()
    conn.close()


def get_liked_songs(user_id):
    conn = get_connection()
    c    = conn.cursor()
    c.execute(
        "SELECT track_name, artists FROM liked_songs WHERE user_id = ? ORDER BY liked DESC",
        (user_id,)
    )
    rows = c.fetchall()
    conn.close()
    return [{"track_name": row["track_name"], "artists": row["artists"]} for row in rows]


def is_liked(user_id, track_name, artists):
    conn = get_connection()
    c    = conn.cursor()
    c.execute(
        "SELECT id FROM liked_songs WHERE user_id = ? AND track_name = ? AND artists = ?",
        (user_id, track_name, artists)
    )
    result = c.fetchone() is not None
    conn.close()
    return result


# Initialise database on import
init_db()