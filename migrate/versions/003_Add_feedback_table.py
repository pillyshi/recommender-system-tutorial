from sqlalchemy import *
from migrate import *


meta = MetaData()

feedback = Table(
    'feedback', meta,
    Column('id', Integer, primary_key=True),
    Column('user_id', Integer),
    Column('item_id', Integer),
    Column('rating', Integer),
    Column('created_at', Date)
)


def upgrade(migrate_engine):
    # Upgrade operations go here. Don't create your own engine; bind
    # migrate_engine to your metadata
    meta.bind = migrate_engine
    feedback.create()


def downgrade(migrate_engine):
    # Operations to reverse the above upgrade go here.
    meta.bind = migrate_engine
    feedback.drop()
