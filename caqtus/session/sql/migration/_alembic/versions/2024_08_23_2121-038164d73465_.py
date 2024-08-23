"""Create tables

Revision ID: 038164d73465
Revises:
Create Date: 2024-08-23 21:21:00.651338

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "038164d73465"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "default_device_configurations",
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("device_type", sa.String(length=255), nullable=False),
        sa.Column("content", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("name"),
    )
    op.create_table(
        "parameters",
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("content", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("name"),
    )
    op.create_table(
        "path",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("path", sa.String(length=255), nullable=False),
        sa.Column("parent_id", sa.Integer(), nullable=True),
        sa.Column("creation_date", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["parent_id"], ["path.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_path_id"), "path", ["id"], unique=False)
    op.create_index(op.f("ix_path_parent_id"), "path", ["parent_id"], unique=False)
    op.create_index(op.f("ix_path_path"), "path", ["path"], unique=True)
    op.create_table(
        "sequences",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("path_id", sa.Integer(), nullable=False),
        sa.Column(
            "state",
            sa.Enum(
                "DRAFT",
                "PREPARING",
                "RUNNING",
                "FINISHED",
                "INTERRUPTED",
                "CRASHED",
                name="state",
            ),
            nullable=False,
        ),
        sa.Column("start_time", sa.DateTime(), nullable=True),
        sa.Column("stop_time", sa.DateTime(), nullable=True),
        sa.Column("expected_number_of_shots", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["path_id"], ["path.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_sequences_path_id"), "sequences", ["path_id"], unique=True)
    op.create_table(
        "sequence.device_configurations",
        sa.Column("id_", sa.Integer(), nullable=False),
        sa.Column("sequence_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("device_type", sa.String(length=255), nullable=False),
        sa.Column("content", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["sequence_id"], ["sequences.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id_"),
        sa.UniqueConstraint("sequence_id", "name", name="device_configuration"),
    )
    op.create_table(
        "sequence.iteration",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("sequence_id", sa.Integer(), nullable=False),
        sa.Column("content", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["sequence_id"], ["sequences.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("sequence_id"),
    )
    op.create_index(
        op.f("ix_sequence.iteration_sequence_id"),
        "sequence.iteration",
        ["sequence_id"],
        unique=False,
    )
    op.create_table(
        "sequence.parameters",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("sequence_id", sa.Integer(), nullable=False),
        sa.Column("content", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["sequence_id"], ["sequences.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("sequence_id"),
    )
    op.create_index(
        op.f("ix_sequence.parameters_sequence_id"),
        "sequence.parameters",
        ["sequence_id"],
        unique=False,
    )
    op.create_table(
        "sequence.time_lanes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("sequence_id", sa.Integer(), nullable=False),
        sa.Column("content", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["sequence_id"], ["sequences.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("sequence_id"),
    )
    op.create_index(
        op.f("ix_sequence.time_lanes_sequence_id"),
        "sequence.time_lanes",
        ["sequence_id"],
        unique=False,
    )
    op.create_table(
        "shots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("sequence_id", sa.Integer(), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("start_time", sa.DateTime(), nullable=False),
        sa.Column("end_time", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["sequence_id"], ["sequences.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("sequence_id", "index", name="shot_identifier"),
    )
    op.create_index(op.f("ix_shots_index"), "shots", ["index"], unique=False)
    op.create_index(
        op.f("ix_shots_sequence_id"), "shots", ["sequence_id"], unique=False
    )
    op.create_table(
        "shot.data.arrays",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("shot_id", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("dtype", sa.String(length=255), nullable=False),
        sa.Column("shape", sa.JSON(), nullable=True),
        sa.Column("bytes", sa.LargeBinary(), nullable=True),
        sa.ForeignKeyConstraint(["shot_id"], ["shots.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("shot_id", "label", name="array_identifier"),
    )
    op.create_index(
        op.f("ix_shot.data.arrays_shot_id"),
        "shot.data.arrays",
        ["shot_id"],
        unique=False,
    )
    op.create_table(
        "shot.data.structured",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("shot_id", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("content", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["shot_id"], ["shots.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("shot_id", "label", name="structured_identifier"),
    )
    op.create_index(
        op.f("ix_shot.data.structured_shot_id"),
        "shot.data.structured",
        ["shot_id"],
        unique=False,
    )
    op.create_table(
        "shot.parameters",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("shot_id", sa.Integer(), nullable=False),
        sa.Column("content", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["shot_id"], ["shots.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_shot.parameters_shot_id"), "shot.parameters", ["shot_id"], unique=True
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f("ix_shot.parameters_shot_id"), table_name="shot.parameters")
    op.drop_table("shot.parameters")
    op.drop_index(
        op.f("ix_shot.data.structured_shot_id"), table_name="shot.data.structured"
    )
    op.drop_table("shot.data.structured")
    op.drop_index(op.f("ix_shot.data.arrays_shot_id"), table_name="shot.data.arrays")
    op.drop_table("shot.data.arrays")
    op.drop_index(op.f("ix_shots_sequence_id"), table_name="shots")
    op.drop_index(op.f("ix_shots_index"), table_name="shots")
    op.drop_table("shots")
    op.drop_index(
        op.f("ix_sequence.time_lanes_sequence_id"), table_name="sequence.time_lanes"
    )
    op.drop_table("sequence.time_lanes")
    op.drop_index(
        op.f("ix_sequence.parameters_sequence_id"), table_name="sequence.parameters"
    )
    op.drop_table("sequence.parameters")
    op.drop_index(
        op.f("ix_sequence.iteration_sequence_id"), table_name="sequence.iteration"
    )
    op.drop_table("sequence.iteration")
    op.drop_table("sequence.device_configurations")
    op.drop_index(op.f("ix_sequences_path_id"), table_name="sequences")
    op.drop_table("sequences")
    op.drop_index(op.f("ix_path_path"), table_name="path")
    op.drop_index(op.f("ix_path_parent_id"), table_name="path")
    op.drop_index(op.f("ix_path_id"), table_name="path")
    op.drop_table("path")
    op.drop_table("parameters")
    op.drop_table("default_device_configurations")
    # ### end Alembic commands ###
