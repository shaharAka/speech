"""Seed the database with built-in Hebrew texts."""

import asyncio
import json
from pathlib import Path

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session_factory, engine
from app.models import Base
from app.models.text import Text
from app.services.model_manager import ensure_base_model


SEED_DIR = Path(__file__).parent.parent / "data" / "seed_texts"


async def migrate_add_round_column(conn) -> None:
    """Add 'round' column to texts table if it doesn't exist (SQLite migration)."""
    result = await conn.execute(text("PRAGMA table_info(texts)"))
    columns = [row[1] for row in result.fetchall()]
    if "round" not in columns:
        await conn.execute(text("ALTER TABLE texts ADD COLUMN round INTEGER NOT NULL DEFAULT 1"))
        print("Added 'round' column to texts table.")
    else:
        print("'round' column already exists.")


async def migrate_add_coaching_columns(conn) -> None:
    """Add coaching_status to training_runs and create coaching_reports table (SQLite migration)."""
    # Add coaching_status to training_runs
    result = await conn.execute(text("PRAGMA table_info(training_runs)"))
    columns = [row[1] for row in result.fetchall()]
    if "coaching_status" not in columns:
        await conn.execute(text("ALTER TABLE training_runs ADD COLUMN coaching_status VARCHAR(20)"))
        print("Added 'coaching_status' column to training_runs table.")

    # Create coaching_reports table if not exists
    await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS coaching_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            training_run_id INTEGER NOT NULL UNIQUE REFERENCES training_runs(id),
            round_number INTEGER NOT NULL,
            next_round_number INTEGER NOT NULL,
            summary_text TEXT NOT NULL,
            insights_json TEXT NOT NULL DEFAULT '[]',
            recommendations_json TEXT NOT NULL DEFAULT '[]',
            wer_trajectory_json TEXT NOT NULL DEFAULT '[]',
            error_analysis_json TEXT NOT NULL DEFAULT '{}',
            difficulty_distribution_json TEXT NOT NULL DEFAULT '{}',
            suggested_next_params_json TEXT,
            texts_generated INTEGER NOT NULL DEFAULT 0,
            is_round1_noise BOOLEAN NOT NULL DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """))
    print("Ensured coaching_reports table exists.")


async def seed_texts(session: AsyncSession) -> None:
    # Check if already seeded
    result = await session.execute(
        select(Text).where(Text.is_builtin == True).limit(1)  # noqa: E712
    )
    if result.scalar_one_or_none():
        print("Texts already seeded, skipping.")
        return

    total = 0
    for json_file in sorted(SEED_DIR.glob("*.json")):
        # Skip macOS resource fork files (._*.json)
        if json_file.name.startswith("._"):
            continue
        with open(json_file, encoding="utf-8") as f:
            texts = json.load(f)

        for t in texts:
            text = Text(
                title=t["title"],
                content=t["content"],
                difficulty=t["difficulty"],
                category="built-in",
                word_count=len(t["content"].split()),
                is_builtin=True,
                round=1,
            )
            session.add(text)
            total += 1

    await session.flush()
    print(f"Seeded {total} texts.")


async def main():
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Migrate: add columns/tables if missing (for existing DBs)
    async with engine.begin() as conn:
        await migrate_add_round_column(conn)
        await migrate_add_coaching_columns(conn)

    async with async_session_factory() as session:
        await seed_texts(session)
        await ensure_base_model(session)
        await session.commit()

    print("Database seeded successfully.")


if __name__ == "__main__":
    asyncio.run(main())
