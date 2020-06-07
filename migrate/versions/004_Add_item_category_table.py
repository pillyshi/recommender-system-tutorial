from sqlalchemy import *
from migrate import *


meta = MetaData()

item_category = Table(
    'item_category', meta,
    Column('id', Integer, primary_key=True),
    Column('item_id', Integer),
    Column('category', String(16))
)

def upgrade(migrate_engine):
    # Upgrade operations go here. Don't create your own engine; bind
    # migrate_engine to your metadata
    meta.bind = migrate_engine
    item_category.create()


def downgrade(migrate_engine):
    # Operations to reverse the above upgrade go here.
    meta.bind = migrate_engine
    item_category.drop()
