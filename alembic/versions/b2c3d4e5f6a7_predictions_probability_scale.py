"""predictions: record which probability scale `confidence` is on

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-09-18 00:00:00.000000

Layer 3 of the diabetes prevalence-correction cleanup.

`predictions.confidence` holds two different quantities depending on when the
row was written. Before the prevalence correction it was the ensemble's raw
output on the 50/50 training prior; after it, the same probability mapped to
the ~14% deployment prevalence. Near the decision threshold the two differ by
roughly a factor of six.

On the development database the six existing diabetes rows sit at 0.90-0.93 —
raw-scale values that would read as extreme risk if interpreted as corrected.

The new nullable column names the scale. Nothing is backfilled: a past row's
scale is not derivable from the row, and guessing it would be worse than
admitting it is unknown. Consumers that aggregate (admin averages, drift
windows) must filter on it rather than assume.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("predictions") as batch:
        batch.add_column(sa.Column("probability_scale", sa.String(16), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("predictions") as batch:
        batch.drop_column("probability_scale")
