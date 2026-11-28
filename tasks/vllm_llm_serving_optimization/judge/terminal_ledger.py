"""Durable sticky lifecycle taint and atomic first-terminal selection."""
import json
import os
from pathlib import Path
import sqlite3
from contextlib import contextmanager

from terminal_contract import canonical, terminal, validate_record


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        existed = self.path.exists()
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS lifecycles (
                    identity TEXT PRIMARY KEY, dispatched INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS slots (
                    identity TEXT PRIMARY KEY, lifecycle TEXT NOT NULL REFERENCES lifecycles(identity),
                    terminal TEXT, started INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS cleanups (
                    identity TEXT PRIMARY KEY REFERENCES slots(identity), observation TEXT NOT NULL);
            ''')
        if not existed:
            fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('PRAGMA synchronous=FULL')
            db.execute('PRAGMA foreign_keys=ON')
            with db:
                yield db
        finally:
            db.close()

    def register(self, slot, lifecycle):
        if not isinstance(lifecycle, str) or not lifecycle or len(lifecycle.encode()) > 512:
            raise ValueError('bounded lifecycle identity required')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT OR IGNORE INTO lifecycles(identity) VALUES (?)', (lifecycle,))
            db.execute('INSERT OR IGNORE INTO slots(identity,lifecycle) VALUES (?,?)', (slot.key(), lifecycle))
            actual, = db.execute('SELECT lifecycle FROM slots WHERE identity=?', (slot.key(),)).fetchone()
            if actual != lifecycle:
                raise ValueError('slot cannot change lifecycle')

    def mark_dispatched(self, slot):
        # FULL-sync commit completes before the caller is allowed to send input.
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT lifecycle,terminal,started FROM slots WHERE identity=?', (slot.key(),)).fetchone()
            if row is None:
                raise ValueError('unregistered slot')
            if row[1] is not None or row[2]:
                return False
            db.execute('UPDATE lifecycles SET dispatched=1 WHERE identity=?', (row[0],))
            db.execute('UPDATE slots SET started=1 WHERE identity=?', (slot.key(),))
        return True

    def dispatched(self, slot):
        with self.connect() as db:
            row = db.execute('SELECT dispatched FROM lifecycles JOIN slots ON lifecycle=lifecycles.identity WHERE slots.identity=?', (slot.key(),)).fetchone()
        if row is None:
            raise ValueError('unregistered slot')
        return bool(row[0])

    def read(self, slot):
        with self.connect() as db:
            row = db.execute('SELECT terminal FROM slots WHERE identity=?', (slot.key(),)).fetchone()
        if row is None:
            raise ValueError('unregistered slot')
        return None if row[0] is None else json.loads(row[0])

    def commit_once(self, slot, record):
        validate_record(record)
        if canonical(record.get('slot')) != slot.key().encode():
            raise ValueError('terminal belongs to another slot')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT terminal,lifecycles.dispatched FROM slots JOIN lifecycles ON slots.lifecycle=lifecycles.identity WHERE slots.identity=?', (slot.key(),)).fetchone()
            if row is None:
                raise ValueError('unregistered slot')
            if row[0] is None:
                if 'candidate_input_dispatched' in record and record['candidate_input_dispatched'] != bool(row[1]):
                    raise ValueError('terminal cannot rewrite durable dispatch observation')
                stored = {**record, 'candidate_input_dispatched': bool(row[1])}
                db.execute('UPDATE slots SET terminal=? WHERE identity=? AND terminal IS NULL', (canonical(stored).decode(), slot.key()))
                return True, stored
            return False, json.loads(row[0])

    def record_cleanup(self, slot, observation):
        if (not isinstance(observation, dict) or set(observation) != {'confirmed', 'errors'}
                or type(observation['confirmed']) is not bool or not isinstance(observation['errors'], list)
                or len(observation['errors']) > 5 or any(error not in {'reap', 'selector_close', 'stdin_close', 'stdout_close', 'stderr_close'} for error in observation['errors'])):
            raise ValueError('invalid cleanup observation')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT terminal FROM slots WHERE identity=?', (slot.key(),)).fetchone()
            if row is None or row[0] is None:
                raise ValueError('cleanup cannot precede primary terminal commitment')
            db.execute('INSERT OR IGNORE INTO cleanups(identity,observation) VALUES (?,?)', (slot.key(), canonical(observation).decode()))

    def cleanup(self, slot):
        with self.connect() as db:
            row = db.execute('SELECT observation FROM cleanups WHERE identity=?', (slot.key(),)).fetchone()
        return None if row is None else json.loads(row[0])

    def unresolved(self):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute('SELECT identity FROM slots WHERE terminal IS NULL ORDER BY identity')]

    def close_unresolved(self, slots, *, elapsed, cap):
        results = []
        for slot in slots:
            existing = self.read(slot)
            if existing is None:
                record = terminal(slot, status='runtime_error', cause='collector_failure',
                                  elapsed=elapsed, cap=cap)
                existing = self.commit_once(slot, record)[1]
            results.append(existing)
        return results
