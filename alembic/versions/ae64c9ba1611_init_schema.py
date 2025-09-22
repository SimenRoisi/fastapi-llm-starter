"""init schema

Revision ID: ae64c9ba1611
Revises: 
Create Date: 2025-08-29 15:12:36.469100

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "init_schema_001"
down_revision = None
branch_labels = None
depends_on = None



def upgrade():
    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("email", sa.String, nullable=False, unique=True),
        sa.Column("api_key", sa.String, nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    op.create_table(
        "api_usage",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("api_key", sa.String, nullable=False),
        sa.Column("endpoint", sa.String, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["api_key"], ["users.api_key"]),
    )

    op.create_index("ix_api_usage_api_key", "api_usage", ["api_key"])
    op.create_index("ix_api_usage_endpoint", "api_usage", ["endpoint"])

def downgrade():
    op.drop_index("ix_api_usage_endpoint", table_name="api_usage")
    op.drop_index("ix_api_usage_api_key", table_name="api_usage")
    op.drop_table("api_usage")
    op.drop_table("users")