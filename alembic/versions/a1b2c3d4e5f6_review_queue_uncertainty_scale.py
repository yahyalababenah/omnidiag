"""review_queue: record the scale and threshold an uncertainty_score was computed on

Revision ID: a1b2c3d4e5f6
Revises: f1e2d3c4b5a6
Create Date: 2026-09-18 00:00:00.000000

Layer 2 of the diabetes prevalence-correction cleanup.

`uncertainty_score` used to be binary entropy centred on 0.5, computed from a
raw-scale probability. It is now entropy centred on the disease's actual
decision threshold, computed from the corrected probability. Those are
different quantities in the same column, so the column alone is no longer
self-describing.

Two nullable columns carry the missing context:

    uncertainty_scale   'raw' | 'corrected' — which probability scale
    decision_threshold  the boundary the entropy was measured around

Existing rows keep NULL in both, which is the honest reading: they were
written before the distinction existed and must not be compared with new
ones. Nothing is backfilled, because the original probability scale of a
pre-correction row cannot be recovered from the row itself.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "f1e2d3c4b5a6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("review_queue") as batch:
        batch.add_column(sa.Column("uncertainty_scale", sa.String(16), nullable=True))
        batch.add_column(sa.Column("decision_threshold", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("review_queue") as batch:
        batch.drop_column("decision_threshold")
        batch.drop_column("uncertainty_scale")
