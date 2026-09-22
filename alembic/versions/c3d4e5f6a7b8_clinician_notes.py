"""predictions + review_queue: store the clinician's own free text

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-09-23 00:00:00.000000

Two columns for the same defect: the product asked clinicians to type free
text and then threw it away.

`predictions.notes` — the doctor's note about a screening. The only free-text
box in the whole UI was the notes parser, whose text was parsed into features
and discarded; nothing was ever stored, so History and the exported PDF had
no record of what the clinician actually thought.

`review_queue.notes` — the annotator's note. POST /api/v4/review/{id}/annotate
has always accepted a `notes` field in its request body and silently dropped
it, because there was no column to put it in. The reviewer's reasoning for a
label is the most useful part of an active-learning annotation and it was the
one part not kept.

Neither column is ever a model input. They are recorded, shown back and
exported; they do not enter `input_features`, the retraining set, or the
DeepSeek report payload.

Nothing is backfilled — the discarded text is gone.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, Sequence[str], None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("predictions", sa.Column("notes", sa.Text(), nullable=True))
    op.add_column("review_queue", sa.Column("notes", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("review_queue", "notes")
    op.drop_column("predictions", "notes")
