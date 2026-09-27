"""predictions: the conformal decision and its Venn-Abers interval

Gate 8.4. A conformal module answers with a decision (referral / no_referral /
uncertain) and an interval around the probability, not with a probability
against a threshold. Both were computed and returned but never stored, so a
prediction record could not be reviewed on the terms the model actually decided
on -- and the interval is what D-32 says replaces risk bands for this model.

Additive and nullable: every existing row keeps NULL and stays valid, and a
threshold module (diabetes) legitimately leaves all three empty.

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
"""
import sqlalchemy as sa
from alembic import op

revision = "e5f6a7b8c9d0"
down_revision = "d4e5f6a7b8c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("predictions", sa.Column("decision", sa.String(length=32), nullable=True))
    op.add_column("predictions", sa.Column("probability_lower", sa.Float(), nullable=True))
    op.add_column("predictions", sa.Column("probability_upper", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("predictions", "probability_upper")
    op.drop_column("predictions", "probability_lower")
    op.drop_column("predictions", "decision")
