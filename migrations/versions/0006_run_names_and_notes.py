"""run names and notes

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-12 09:00:00
"""
from alembic import op
import sqlalchemy as sa


revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None


def upgrade():
    # Defaults on the server side, so runs saved before this have them too.
    with op.batch_alter_table('runs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('name', sa.String(length=200),
                                      nullable=False, server_default=''))
        batch_op.add_column(sa.Column('note', sa.Text(), nullable=False,
                                      server_default=''))


def downgrade():
    with op.batch_alter_table('runs', schema=None) as batch_op:
        batch_op.drop_column('note')
        batch_op.drop_column('name')
