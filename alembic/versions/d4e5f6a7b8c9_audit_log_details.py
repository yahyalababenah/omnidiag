"""audit_logs.details — JSON for request facts the path does not carry

Some requests are not auditable from the request line alone. A heart batch
upload is read under a declared chest-pain coding (Gate 8.2), and a batch whose
coding was never recorded cannot be rechecked afterwards. The alternative —
putting the coding in the URL — would have polluted the `endpoint` grouping the
monitoring dashboards key on.

Additive and nullable: every existing row keeps NULL and stays valid.

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
"""
import sqlalchemy as sa
from alembic import op

revision = "d4e5f6a7b8c9"
down_revision = "c3d4e5f6a7b8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("audit_logs", sa.Column("details", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("audit_logs", "details")
