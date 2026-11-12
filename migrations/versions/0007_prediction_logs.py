"""prediction logs

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-12 10:00:00
"""
from alembic import op
import sqlalchemy as sa


revision = '0007'
down_revision = '0006'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('prediction_logs',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('project_id', sa.String(length=32), nullable=False),
    sa.Column('version_id', sa.String(length=32), nullable=True),
    sa.Column('model_name', sa.String(length=200), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('stage', sa.String(length=16), nullable=False),
    sa.Column('source', sa.String(length=16), nullable=False),
    sa.Column('caller', sa.String(length=200), nullable=False),
    sa.Column('rows', sa.Integer(), nullable=False),
    sa.Column('flagged', sa.Integer(), nullable=False),
    sa.Column('latency_ms', sa.Integer(), nullable=False),
    sa.Column('object_key', sa.String(length=500), nullable=False),
    sa.Column('at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'],
                            ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['version_id'], ['model_versions.id'],
                            ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('prediction_logs', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_prediction_logs_project_id'),
                              ['project_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_prediction_logs_at'), ['at'],
                              unique=False)


def downgrade():
    with op.batch_alter_table('prediction_logs', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_prediction_logs_at'))
        batch_op.drop_index(batch_op.f('ix_prediction_logs_project_id'))
    op.drop_table('prediction_logs')
