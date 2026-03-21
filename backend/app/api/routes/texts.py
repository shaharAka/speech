import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.recording import Recording
from app.models.text import Text
from app.schemas.text import (
    GenerateRoundResponse,
    RoundProgressResponse,
    TextCreate,
    TextListResponse,
    TextResponse,
    TextUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _count_words(content: str) -> int:
    return len(content.split())


def _text_to_response(t, rec_count: int = 0) -> TextResponse:
    return TextResponse(
        id=t.id,
        title=t.title,
        content=t.content,
        difficulty=t.difficulty,
        category=t.category,
        word_count=t.word_count,
        is_builtin=t.is_builtin,
        round=t.round,
        created_at=t.created_at,
        recording_count=rec_count,
    )


@router.get("/round-progress", response_model=RoundProgressResponse)
async def round_progress(db: AsyncSession = Depends(get_db)):
    """Get progress for the current (latest) round."""
    # Find current round
    max_round_result = await db.execute(select(func.max(Text.round)))
    current_round = max_round_result.scalar_one() or 1

    # Count total texts in current round
    total_result = await db.execute(
        select(func.count(Text.id)).where(Text.round == current_round)
    )
    total_texts = total_result.scalar_one()

    # Count texts with at least 1 recording in current round
    practiced_result = await db.execute(
        select(func.count(func.distinct(Text.id)))
        .select_from(Text)
        .join(Recording, Recording.text_id == Text.id)
        .where(Text.round == current_round)
    )
    practiced_texts = practiced_result.scalar_one()

    is_complete = total_texts > 0 and practiced_texts >= total_texts

    # Include performance summary when round is complete
    performance_summary = None
    if is_complete:
        from app.services.text_generator import analyze_round_performance
        performance_summary = await analyze_round_performance(current_round, db)

    return RoundProgressResponse(
        current_round=current_round,
        total_texts=total_texts,
        practiced_texts=practiced_texts,
        is_complete=is_complete,
        performance_summary=performance_summary,
    )


@router.post("/generate-round", response_model=GenerateRoundResponse)
async def generate_round(db: AsyncSession = Depends(get_db)):
    """Generate the next round of adaptive texts using Gemini."""
    from app.services.text_generator import create_next_round

    # Check current round is complete
    max_round_result = await db.execute(select(func.max(Text.round)))
    current_round = max_round_result.scalar_one() or 1

    total_result = await db.execute(
        select(func.count(Text.id)).where(Text.round == current_round)
    )
    total_texts = total_result.scalar_one()

    practiced_result = await db.execute(
        select(func.count(func.distinct(Text.id)))
        .select_from(Text)
        .join(Recording, Recording.text_id == Text.id)
        .where(Text.round == current_round)
    )
    practiced_texts = practiced_result.scalar_one()

    if total_texts > 0 and practiced_texts < total_texts:
        raise HTTPException(
            status_code=400,
            detail=f"Current round not complete yet ({practiced_texts}/{total_texts} texts practiced)",
        )

    try:
        result = await create_next_round(db)
        await db.commit()
        return GenerateRoundResponse(
            round=result["round"],
            texts_created=result["texts_created"],
        )
    except Exception as e:
        logger.error(f"Failed to generate round: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate texts: {str(e)}")


@router.get("", response_model=TextListResponse)
async def list_texts(
    difficulty: str | None = None,
    category: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    query = select(Text)
    count_query = select(func.count(Text.id))

    if difficulty:
        query = query.where(Text.difficulty == difficulty)
        count_query = count_query.where(Text.difficulty == difficulty)
    if category:
        query = query.where(Text.category == category)
        count_query = count_query.where(Text.category == category)

    query = query.order_by(Text.created_at.desc()).limit(limit).offset(offset)

    result = await db.execute(query)
    texts = list(result.scalars().all())

    count_result = await db.execute(count_query)
    total = count_result.scalar_one()

    # Get recording counts
    items = []
    for t in texts:
        rec_count_result = await db.execute(
            select(func.count(Recording.id)).where(Recording.text_id == t.id)
        )
        rec_count = rec_count_result.scalar_one()
        items.append(_text_to_response(t, rec_count))

    return TextListResponse(items=items, total=total)


@router.get("/{text_id}", response_model=TextResponse)
async def get_text(text_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Text).where(Text.id == text_id))
    text = result.scalar_one_or_none()
    if not text:
        raise HTTPException(status_code=404, detail="Text not found")

    rec_count_result = await db.execute(
        select(func.count(Recording.id)).where(Recording.text_id == text.id)
    )
    rec_count = rec_count_result.scalar_one()

    return _text_to_response(text, rec_count)


@router.post("", response_model=TextResponse, status_code=201)
async def create_text(body: TextCreate, db: AsyncSession = Depends(get_db)):
    # Get current round for custom texts
    max_round_result = await db.execute(select(func.max(Text.round)))
    current_round = max_round_result.scalar_one() or 1

    text = Text(
        title=body.title,
        content=body.content,
        difficulty=body.difficulty,
        category="custom",
        word_count=_count_words(body.content),
        is_builtin=False,
        round=current_round,
    )
    db.add(text)
    await db.flush()

    return _text_to_response(text, 0)


@router.put("/{text_id}", response_model=TextResponse)
async def update_text(text_id: int, body: TextUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Text).where(Text.id == text_id))
    text = result.scalar_one_or_none()
    if not text:
        raise HTTPException(status_code=404, detail="Text not found")
    if text.is_builtin:
        raise HTTPException(status_code=400, detail="Cannot edit built-in texts")

    if body.title is not None:
        text.title = body.title
    if body.content is not None:
        text.content = body.content
        text.word_count = _count_words(body.content)
    if body.difficulty is not None:
        text.difficulty = body.difficulty

    rec_count_result = await db.execute(
        select(func.count(Recording.id)).where(Recording.text_id == text.id)
    )
    rec_count = rec_count_result.scalar_one()

    return _text_to_response(text, rec_count)


@router.delete("/{text_id}", status_code=204)
async def delete_text(text_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Text).where(Text.id == text_id))
    text = result.scalar_one_or_none()
    if not text:
        raise HTTPException(status_code=404, detail="Text not found")
    if text.is_builtin:
        raise HTTPException(status_code=400, detail="Cannot delete built-in texts")

    await db.delete(text)
