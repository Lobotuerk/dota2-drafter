import pytest
from dota2drafter.processor.hero_indexer import HeroIndexer


def test_hero_indexer():
    indexer = HeroIndexer()
    
    heroes = [
        {"id": 1, "name": "Anti-Mage", "playable": True},
        {"id": 2, "name": "Axe", "playable": False},  # Non-playable
        {"id": 4, "name": "Bloodseeker", "playable": True},
        {"id": 3, "name": "Bane", "playable": True},
    ]
    
    indexer.build_mapping(heroes)
    
    # Non-playable should be excluded, and indices should be sorted by API hero ID
    # Sorted order of playable ids: 1, 3, 4
    # Expected contiguous index mappings:
    # 1 -> 1
    # 3 -> 2
    # 4 -> 3
    
    assert indexer.get_contiguous_count() == 3
    
    assert indexer.map_hero_id(1) == 1
    assert indexer.map_hero_id(3) == 2
    assert indexer.map_hero_id(4) == 3
    assert indexer.map_hero_id(2) is None  # Non-playable is not mapped
    assert indexer.map_hero_id(99) is None  # Unknown hero id
    
    assert indexer.get_mapping() == {1: 1, 3: 2, 4: 3}
